//===--- AddressLowering.cpp - Lower SIL address-only types. --------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2017 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
// This pass lowers SILTypes. On completion, the SILType of every SILValue is
// its SIL storage type. A SIL storage type is always an address type for values
// that require indirect storage at the LLVM IR level. Consequently, this pass
// is required for IRGen. It is a mandatory IRGen preparation pass (not a
// diagnostic pass).
//
// The SIL input must already be ownership-lowered such that semantic copies do
// not produce new SIL values. i.e. copy_value/destroy_value instructions are
// not allowed. Instead we expect release_value/retain_value. The
// OwnershipElimination pass handles this lowering.
//
// In the following comments, items marked "[REUSE]" only apply to the proposed
// storage reuse optimization, which is not currently implemented. Note: We
// probably won't need to implement the [REUSE] optimizations at all. The
// currently implemented SSA-based on-the-fly optimization and
// BlockArgumentStorageOptimizer already record storage projections and are
// doing a good job. LLVM should handle the stack coloring aspect.
//
// ## State
//
// A `valueStorageMap` maps each opaque SIL value to its storage
// information containing:
//
// - An ordinal representing the position of this instruction.
//
// - [REUSE] The identifier of the storage object. An optimized storage object
//   may have multiple disjoint lifetimes. A storage object may also have
//   subobjects. Each subobject has its own live range. When considering
//   liveness of the subobject, one must also consider liveness of the
//   parent object.
//
// - If this is a subobject projection, refer back to the value whose
//  storage object will be the parent that this storage address is a
//  projection of.
//
// - The storage address for this subobject.
//
// Note: During all steps below, the values referenced in the valueStorageMap
// must remain valid. Each storage map entry refers to a valid value, with the
// same number of operands. Storage map entries that refer to function arguments
// or block arguments may be replaced with loads without violating this
// invariant. After the final step, all original address-only values are
// deleted.
//
// ## Step #1: Map opaque values
//
// Populate `valueStorageMap` in forward order (RPO), giving each opaque value
// an ordinal position.
//
// [REUSE] Assign a storage identifier to each opaque value. Optionally optimize
// storage by assigning multiple values the same identifier.
//
// ## Step #2: Allocate storage
//
// In reverse order (PO), allocate the parent storage object for each opaque
// value.
//
// [REUSE] If storage has already been allocated for the current live range,
// then simply reuse it.
//
// If the value's use composes a parent object from this value (struct, tuple,
// enum), and use's storage can be projected from, then mark the value's storage
// as a projection from the use value. [REUSE] Also inherit the use's storage
// identifier, and add an interval to the live range with the current projection
// path.
//
// A use can be projected from if its allocation is available at (dominates)
// this value and using the same storage over the interval from this value to
// the use does not overlap with the existing live range.
//
// If the value is a subobject extraction (struct_extract, tuple_extract,
// open_existential_value, unchecked_enum_data), then the def's storage can be
// extracted from. Mark the value's storage as a projection from the def.
//
// Checking interference requires checking all operands that have been marked as
// projections. In the case of block arguments, it means checking the terminator
// operands of all predecessor blocks.
//
// [REUSE] Rather than checking all value operands, each live range will contain
// a set of intervals. Each interval will be associated with a projection path.
//
// Opaque value's that are the root of all projection paths now have their
// `storageAddress` assigned to an `alloc_stack` or argument. Opaque value's
// that are projections do not yet have a `storageAddress`.
//
// ## Step #3. Rewrite opaque values
//
// In forward order (RPO), rewrite each opaque value definition, and all its
// uses. This generally involves creating a new `_addr` variant of the
// instruction and obtaining the storage address from the `valueStorageMap`.
//
// If this value's storage is a projection of the value defined by its composing
// use, then first generate instructions to materialize the projection. This is
// a recursive process starting with the root of the projection path.
//
// A projection path will be materialized once, for the leaf subobject. When
// this happens, the `storageAddress` will be assigned for any intermediate
// projection paths. When those values are rewritten, their `storageAddress`
// will already be available.
//
// TODO: Call result rewriting needs a lot of cleanup. The right way to go about
// it is first to implement multi-result calls, then almost all the horrible
// hackery can be stripped away. See TODO: MultiValue.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "address-lowering"
#include "swift/Basic/Range.h"
#include "swift/SIL/BasicBlockUtils.h"
#include "swift/SIL/DebugUtils.h"
#include "swift/SIL/PrettyStackTrace.h"
#include "swift/SIL/SILArgument.h"
#include "swift/SIL/SILBuilder.h"
#include "swift/SIL/SILVisitor.h"
#include "swift/SILOptimizer/Analysis/PostOrderAnalysis.h"
#include "swift/SILOptimizer/PassManager/Transforms.h"
#include "swift/SILOptimizer/Utils/Local.h"
#include "swift/SILOptimizer/Utils/StackNesting.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace swift;
using llvm::SmallSetVector;
using llvm::PointerIntPair;

llvm::cl::opt<bool>
    OptimizeOpaqueAddressLowering("optimize-opaque-address-lowering",
                                  llvm::cl::init(true));

namespace {
struct GetUser {
  SILInstruction *operator()(Operand *oper) const { return oper->getUser(); }
};
} // namespace

// TODO: LLVM needs a map_range.
iterator_range<llvm::mapped_iterator<ValueBase::use_iterator, GetUser>>
getUserRange(SILValue val) {
  return make_range(llvm::map_iterator(val->use_begin(), GetUser()),
                    llvm::map_iterator(val->use_end(), GetUser()));
}

// Visit all "actual" call results.
// Stop when the visitor returns `false`.
//
// TODO: MultiValue. Remove this helper.
static void visitCallResults(ApplySite apply,
                             llvm::function_ref<bool(SILValue)> visitor) {
  // FIXME: this entire implementation only really works for ApplyInst.
  auto applyInst = cast<ApplyInst>(apply);
  if (applyInst->getType().is<TupleType>()) {
    for (auto *operand : applyInst->getUses()) {
      if (auto extract = dyn_cast<TupleExtractInst>(operand->getUser()))
        if (!visitor(extract))
          break;
    }
  } else
    visitor(applyInst);
}

// Get the argument of a try_apply, or nullptr if it is unused.
//
// TODO: MultiValue: Only relevant for tuple pseudo results.
static SILPHIArgument *getTryApplyPseudoResult(TryApplyInst *TAI) {
  auto *argBB = TAI->getNormalBB();
  assert(argBB->getNumArguments() == 1);
  return argBB->getPHIArguments()[0];
}

// Get the tuple pseudo-value returned by the call.
//
// TODO: MultiValue: Only relevant for tuple pseudo results.
SILValue getCallPseudoResult(ApplySite apply) {
  return isa<ApplyInst>(apply)
             ? SILValue(cast<SingleValueInstruction>(apply.getInstruction()))
             : SILValue(getTryApplyPseudoResult(cast<TryApplyInst>(apply)));
}

// TODO: MultiValue. Calls are SILValues, but when the result type is a tuple,
// the call value does not represent a real value with storage. This is a
// horrible situation for address lowering because there's no way to tell from
// any given value whether its legal to assign storage to that
// value. Consequently, the implementation of call lowering is a house of cards
// that doesn't fall out naturally from the algorithm that lowers values to
// storage.
static bool isPseudoCallResult(SILValue value) {
  if (isa<ApplyInst>(value))
    return value->getType().is<TupleType>();

  auto *bbArg = dyn_cast<SILPHIArgument>(value);
  if (!bbArg)
    return false;

  auto *predBB = bbArg->getParent()->getSinglePredecessorBlock();
  if (!predBB)
    return false;

  return isa<TryApplyInst>(predBB->getTerminator())
         && bbArg->getType().is<TupleType>();
}

/// Return the SILValue that represents the "user" of the given operand. If
/// the operand's use is a SingleValueInstruction, the value is the user
/// itself. If the operand's use is a block terminator, it is the
/// corresponding block argument that represents the operand's value.
///
/// For multi-value instructions, return an invalid SILValue. The caller will
/// need to deal with those.
///
/// TODO: Make this a utility or at least add
/// CondBranchInst::getBlockArgForOperand?
SILValue getValueOfOperandUse(Operand *operand) {
  SILInstruction *user = operand->getUser();
  auto *singleVal = dyn_cast<SingleValueInstruction>(user);
  if (singleVal)
    return singleVal;

  // The caller needs to handle multi-value instructions.
  // FIXME: MultiValue.
  assert(isa<TermInst>(user));

  switch (user->getKind()) {
  default:
#ifndef NDEBUG
    user->dump();
#endif
    llvm_unreachable("Unexpected block terminator with address-only operand.");

  case SILInstructionKind::BranchInst: {
    auto *BI = cast<BranchInst>(user);
    unsigned bbArgIdx = BI->getArgIndexOfOperand(operand->getOperandNumber());
    return BI->getDestBB()->getArgument(bbArgIdx);
  }
  case SILInstructionKind::CondBranchInst: {
    auto *CBI = cast<CondBranchInst>(user);
    unsigned opIdx = operand->getOperandNumber();
    unsigned bbArgIdx;
    if (CBI->isTrueOperandIndex(opIdx)) {
      bbArgIdx = opIdx - CBI->getTrueOperands()[0].getOperandNumber();
      return CBI->getTrueBB()->getArgument(bbArgIdx);
    } else {
      assert(CBI->isFalseOperandIndex(opIdx));
      bbArgIdx = opIdx - CBI->getFalseOperands()[0].getOperandNumber();
      return CBI->getFalseBB()->getArgument(bbArgIdx);
    }
  }
  }
}

//===----------------------------------------------------------------------===//
// ValueStorageMap: Map Opaque/Resilient SILValues to abstract storage units.
//===----------------------------------------------------------------------===//

namespace {
// Avoid storing operand references in case some def-use rewritting occurs
// before formal rewritting.
struct ValueStorage {
  /// The final address of this storage unit after rewriting the SIL.
  /// For values linked to their own storage, this is set during storage
  /// allocation. For projections, it is only set after instruction rewriting.
  SILValue storageAddress;
  uint32_t projectedStorageID;
  uint16_t projectedOperandNum;
  unsigned isUseProjection : 1;
  unsigned isDefProjection : 1;
  unsigned isRewritten : 1;

  ValueStorage() { clear(); }

  void clear() {
    storageAddress = SILValue();
    projectedStorageID = ~0;
    projectedOperandNum = ~0;
    isUseProjection = false;
    isDefProjection = false;
    isRewritten = false;
  }

  bool isAllocated() const {
    return storageAddress || isUseProjection || isDefProjection;
  }

  void markRewritten() {
    assert(storageAddress);
    isRewritten = true;
  }
};

/// Map each opaque/resilient SILValue to its abstract storage.
/// O(1) membership test.
/// O(n) iteration in RPO order.
///
/// This doesn't have an eraseValue because it values are expected to be created
/// at once in RPO order so that instructions are succesfully deleted later.
class ValueStorageMap {
  struct ValueStoragePair {
    SILValue value;
    ValueStorage storage;
    ValueStoragePair(SILValue v, ValueStorage s) : value(v), storage(s) {}
  };
  typedef std::vector<ValueStoragePair> ValueVector;
  // Hash of values to ValueVector indices.
  typedef llvm::DenseMap<SILValue, unsigned> ValueHashMap;

  ValueVector valueVector;
  ValueHashMap valueHashMap;

public:
  bool empty() const { return valueVector.empty(); }

  void clear() {
    valueVector.clear();
    valueHashMap.clear();
  }

  // Iterate over value storage. Once we begin erasing instructions, some
  // entries could become invalid. ValueStorage validity can be checked with
  // valueStorageMap.contains(value).
  ValueVector::iterator begin() { return valueVector.begin(); }

  ValueVector::iterator end() { return valueVector.end(); }

  ValueVector::reverse_iterator rbegin() { return valueVector.rbegin(); }

  ValueVector::reverse_iterator rend() { return valueVector.rend(); }

  bool contains(SILValue value) const {
    return valueHashMap.find(value) != valueHashMap.end();
  }

  unsigned getOrdinal(SILValue value) const {
    auto hashIter = valueHashMap.find(value);
    assert(hashIter != valueHashMap.end() && "Missing SILValue");
    return hashIter->second;
  }

  ValueStorage &getStorage(SILValue value) {
    return valueVector[getOrdinal(value)].storage;
  }
  const ValueStorage &getStorage(SILValue value) const {
    return valueVector[getOrdinal(value)].storage;
  }

  // This must be called in RPO order.
  ValueStorage &insertValue(SILValue value) {
    auto hashResult =
        valueHashMap.insert(std::make_pair(value, valueVector.size()));
    (void)hashResult;
    assert(hashResult.second && "SILValue already mapped");

    valueVector.emplace_back(value, ValueStorage());

    return valueVector.back().storage;
  }

  void replaceValue(SILValue oldValue, SILValue newValue) {
    auto pos = valueHashMap.find(oldValue);
    assert(pos != valueHashMap.end());
    unsigned ordinal = pos->second;
    valueHashMap.erase(pos);

    auto hashResult = valueHashMap.insert(std::make_pair(newValue, ordinal));
    (void)hashResult;
    assert(hashResult.second && "SILValue already mapped");

    valueVector[ordinal].value = newValue;
  }

  /// Follow one level of storage projection. The returned storage may also be a
  /// projection.
  ValueStoragePair &getProjectedStorage(ValueStorage &storage) {
    assert(storage.isUseProjection || storage.isDefProjection);
    return valueVector[storage.projectedStorageID];
  }

  /// Record a storage projection from the use of the given operand
  /// (e.g. struct, tuple, enum) into the operand's source.
  void setComposedUseProjection(Operand *oper) {
    auto &storage = getStorage(oper->get());
    storage.projectedStorageID = getOrdinal(getValueOfOperandUse(oper));
    storage.projectedOperandNum = oper->getOperandNumber();
    storage.isUseProjection = true;
  }

  /// Record a storage projection from the source of the given operand into its
  /// use (e.g. struct_extract, tuple_extract).
  void setExtractedDefOperand(Operand *oper) {
    assert(isa<SingleValueInstruction>(oper->getUser()));
    auto &storage = getStorage(oper->get());
    storage.projectedStorageID = getOrdinal(oper->get());
    storage.projectedOperandNum = oper->getOperandNumber();
    storage.isDefProjection = true;
  }

  /// Return true if the given operand projects storage from its use into its
  /// source.
  bool isUseProjection(Operand *oper) const {
    auto &srcStorage = getStorage(oper->get());
    if (!srcStorage.isUseProjection)
      return false;

    return srcStorage.projectedOperandNum == oper->getOperandNumber();
  }

  // Given storage for a value, return the operand of the value's use
  // (e.g. struct, tuple, enum) that projects storage from the use to the
  // operand's source.
  //
  // Postcondition: valueStorageMap.getStorage(result->get()) == storage.
  Operand *getUseProjectionOperand(ValueStorage &storage) {
    assert(storage.isUseProjection);
    SILValue useVal = getProjectedStorage(storage).value;
    auto *useInst = useVal->getDefiningInstruction();
    return &useInst->getAllOperands()[storage.projectedOperandNum];
  }

  // Given storage for a value (e.g. defined by struct_extract, tuple_extract),
  // return the operand of the value's defining instruction that projects
  // storage from the operands source value.
  //
  // Postcondition:
  //   valueStorageMap.getStorage(getValueOfOperandUse(result)) == storage.
  Operand *getDefProjectionOperand(SingleValueInstruction *extractInst) {
    auto &storage = getStorage(extractInst);
    assert(storage.isDefProjection);
    return &extractInst->getAllOperands()[storage.projectedOperandNum];
  }

#ifndef NDEBUG
  void dump() {
    llvm::dbgs() << "ValueStorageMap:\n";
    for (unsigned ordinal : indices(valueVector)) {
      auto &valStoragePair = valueVector[ordinal];
      llvm::dbgs() << "value: ";
      valStoragePair.value->dump();
      auto &storage = valStoragePair.storage;
      if (storage.isUseProjection) {
        llvm::dbgs() << "  use projection: ";
        if (!storage.isRewritten)
          valueVector[storage.projectedStorageID].value->dump();
      } else if (storage.isDefProjection) {
        llvm::dbgs() << "  def projection: ";
        if (!storage.isRewritten)
          valueVector[storage.projectedStorageID].value->dump();
      }
      if (storage.storageAddress) {
        llvm::dbgs() << "  storage: ";
        storage.storageAddress->dump();
      }
    }
  }
#endif
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// AddressLoweringState: shared state for the pass's analysis and transforms.
//===----------------------------------------------------------------------===//

namespace {
struct AddressLoweringState {
  SILFunction *F;
  SILFunctionConventions loweredFnConv;

  // Dominators remain valid throughout this pass.
  DominanceInfo *domInfo;

  // Map opened archetypes so that AllocStackInst can be created on demand.
  SILOpenedArchetypesTracker openedArchetypesTracker;

  // All opaque values and associated storage.
  ValueStorageMap valueStorageMap;
  // All call sites with formally indirect SILArgument or SILResult conventions.
  SmallSetVector<ApplySite, 16> indirectApplies;
  // All function-exiting terminators (return or throw instructions).
  SmallVector<SILInstruction *, 8> exitingInsts;
  // Delete these instructions after performing transformations. They must
  // already be dead (no remaining users) before being added to this set.
  SmallSetVector<SILInstruction *, 16> instsToDelete;

#ifndef NDEBUG
  // Calls are removed after everything else.
  // FIXME: MultiValue. Until calls are deleted like all other values, they need
  // to be carefully orchestrated w.r.t. their tuple values.
  SmallSetVector<ApplyInst *, 16> callsToDelete;
#endif

  AddressLoweringState(SILFunction *F, DominanceInfo *domInfo)
      : F(F),
        loweredFnConv(F->getLoweredFunctionType(),
                      SILModuleConventions::getLoweredAddressConventions()),
        domInfo(domInfo), openedArchetypesTracker(F) {}

  bool isDead(SILInstruction *inst) const { return instsToDelete.count(inst); }

  void markDead(SILInstruction *inst) {
#ifndef NDEBUG
    for (auto result : inst->getResults())
      for (Operand *use : result->getUses())
        assert(instsToDelete.count(use->getUser()));
#endif
    instsToDelete.insert(inst);
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// OpaqueValueVisitor: Map OpaqueValues to ValueStorage.
//===----------------------------------------------------------------------===//

namespace {
/// Collect all opaque/resilient values, inserting them in `valueStorageMap` in
/// RPO order.
///
/// Collect all call arguments with formally indirect SIL argument convention in
/// `indirectOperands` and formally indirect SIL results in `indirectResults`.
///
/// TODO: Perform linear-scan style in-place stack slot coloring by keeping
/// track of each value's last use.
class OpaqueValueVisitor {
  AddressLoweringState &pass;
  PostOrderFunctionInfo postorderInfo;

public:
  explicit OpaqueValueVisitor(AddressLoweringState &pass)
      : pass(pass), postorderInfo(pass.F) {}

  void mapValueStorage();

protected:
  void checkForIndirectApply(ApplySite applySite);
  void visitValue(SILValue value);
};
} // end anonymous namespace

/// Top-level entry: Populate `valueStorageMap`, `indirectResults`, and
/// `indirectOperands`.
///
/// Find all Opaque/Resilient SILValues and add them
/// to valueStorageMap in RPO.
void OpaqueValueVisitor::mapValueStorage() {
  for (auto *BB : postorderInfo.getReversePostOrder()) {
    if (BB->getTerminator()->isFunctionExiting())
      pass.exitingInsts.push_back(BB->getTerminator());

    // Opaque function arguments have already been replaced.
    if (BB != pass.F->getEntryBlock()) {
      for (auto *arg : BB->getArguments()) {
        if (isPseudoCallResult(arg))
          continue;

        visitValue(arg);
      }
    }
    for (auto &II : *BB) {
      pass.openedArchetypesTracker.registerOpenedArchetypes(&II);

      if (auto apply = ApplySite::isa(&II)) {
        checkForIndirectApply(apply);

        // Do not call visitValue on pseudo call results.
        if (isPseudoCallResult(getCallPseudoResult(apply)))
          continue;
      }
      for (auto result : II.getResults())
        visitValue(result);
    }
  }
}

/// Populate `indirectApplies`.
void OpaqueValueVisitor::checkForIndirectApply(ApplySite applySite) {
  auto calleeConv = applySite.getSubstCalleeConv();
  unsigned calleeArgIdx = applySite.getCalleeArgIndexOfFirstAppliedArg();
  for (Operand &operand : applySite.getArgumentOperands()) {
    if (operand.get()->getType().isObject()) {
      auto argConv = calleeConv.getSILArgumentConvention(calleeArgIdx);
      if (argConv.isIndirectConvention()) {
        pass.indirectApplies.insert(applySite);
        break;
      }
    }
    ++calleeArgIdx;
  }
  if (applySite.getSubstCalleeType()->hasIndirectFormalResults())
    pass.indirectApplies.insert(applySite);
}

/// If `value` is address-only add it to the `valueStorageMap`.
void OpaqueValueVisitor::visitValue(SILValue value) {
  if (value->getType().isObject()
      && value->getType().isAddressOnly(pass.F->getModule())) {
    if (pass.valueStorageMap.contains(value)) {
      assert(ApplySite::isa(value)
             || isa<SILFunctionArgument>(
                    pass.valueStorageMap.getStorage(value).storageAddress));
      return;
    }
    pass.valueStorageMap.insertValue(value);
  }
}

//===----------------------------------------------------------------------===//
// BlockArgumentStorageOptimizer - reuse storage across block arguments.
//===----------------------------------------------------------------------===//

// Populate projectedBBArgs with all inputs to bbArg that can reuse the
// argument's storage.
//
// Blocks are marked white, grey, or black.
//
// All blocks start white.
// Set all predecessor blocks black.
//
// For each incoming value:
//
//   Mark the current predecessor white (from black) if it is *not* a critical
//   edge.  We know no other source will be live out of that predecessor because
//   this block will be marked black when we process the other incoming values.
//
//   For all uses: scan the CFG backward following predecessors.
//     If the current block is:
//     White: mark it grey and continue scanning.
//     Grey: stop scanning and continue with the next use.
//     Black: record an interference, stop scanning, continue with the next use.
//
//   If no black blocks were reached, record this incoming value as a valid
//   projection.
//
//   Mark all grey blocks black. This will mark the incoming predecessor black
//   again, along with any other blocks in which the incoming value is live-out.
//
// In the end, we have a set of non-interfering incoming values that can reuse
// the bbArg's storage.
namespace {
class BlockArgumentStorageOptimizer {
  class Result {
    friend class BlockArgumentStorageOptimizer;
    SmallVector<Operand *, 4> projectedBBArgs;

    struct GetOper {
      SILValue operator()(Operand *oper) const { return oper->get(); }
    };

  public:
    ArrayRef<Operand *> getArgumentProjections() const {
      return projectedBBArgs;
    }

    // TODO: LLVM needs a map_range.
    iterator_range<
        llvm::mapped_iterator<ArrayRef<Operand *>::iterator, GetOper>>
    getIncomingValueRange() const {
      return make_range(
          llvm::map_iterator(getArgumentProjections().begin(), GetOper()),
          llvm::map_iterator(getArgumentProjections().end(), GetOper()));
    }

    void clear() { projectedBBArgs.clear(); }
  };

  SILPHIArgument *bbArg;
  Result result;

  // Working state for this bbArg.
  //
  // TODO: These are possible candidates for bitsets since we're reusing storage
  // across multiple uses and want to perform a fast union.
  SmallPtrSet<SILBasicBlock *, 16> blackBlocks;
  SmallPtrSet<SILBasicBlock *, 16> greyBlocks;

  // Working state per-incoming-value.
  SmallVector<SILBasicBlock *, 16> liveBBWorklist;

public:
  BlockArgumentStorageOptimizer(SILPHIArgument *bbArg) : bbArg(bbArg) {}

  Result &&computeArgumentProjections() &&;

protected:
  bool computeIncomingLiveness(Operand *useOper, SILBasicBlock *defBB);
};
} // namespace

// Process an incoming value.
//
// Fully compute liveness from this use operand. Return true if no interference
// was detected along the way.
bool BlockArgumentStorageOptimizer::computeIncomingLiveness(
    Operand *useOper, SILBasicBlock *defBB) {
  bool noInterference = true;

  auto visitLiveBlock = [&](SILBasicBlock *liveBB) {
    if (blackBlocks.count(liveBB))
      noInterference = false;
    else if (greyBlocks.insert(liveBB).second && liveBB != defBB)
      liveBBWorklist.push_back(liveBB);
  };

  assert(liveBBWorklist.empty());

  visitLiveBlock(useOper->getUser()->getParent());

  while (!liveBBWorklist.empty()) {
    auto *succBB = liveBBWorklist.pop_back_val();
    for (auto *predBB : succBB->getPredecessorBlocks())
      visitLiveBlock(predBB);
  }
  return noInterference;
}

// Process this bbArg, recording in the Result which incoming values can reuse
// storage with the argument itself.
BlockArgumentStorageOptimizer::Result &&
BlockArgumentStorageOptimizer::computeArgumentProjections() && {
  SmallVector<Operand *, 4> incomingOperands;
  bbArg->getIncomingOperands(incomingOperands);

  // Prune the single incoming value case.
  if (incomingOperands.size() == 1) {
    result.projectedBBArgs.push_back(incomingOperands[0]);
    return std::move(result);
  }

  SILBasicBlock *succBB = bbArg->getParent();
  for (auto *predBB : succBB->getPredecessorBlocks()) {
    // Disallow block arguments on critical edges.
    assert(predBB->getSingleSuccessorBlock() == succBB);
    blackBlocks.insert(predBB);
  }

  for (auto *incomingOper : incomingOperands) {
    SILBasicBlock *incomingPred = incomingOper->getUser()->getParent();
    SILValue incomingVal = incomingOper->get();

    bool erased = blackBlocks.erase(incomingPred);
    (void)erased;
    assert(erased);

    bool noInterference = true;
    // Continue marking live blocks even after detecting an interference so that
    // the live set is complete when evaluating subsequent incoming vales.
    for (auto *use : incomingVal->getUses()) {
      noInterference &=
          computeIncomingLiveness(use, incomingVal->getParentBlock());
    }
    if (noInterference)
      result.projectedBBArgs.push_back(incomingOper);

    blackBlocks.insert(greyBlocks.begin(), greyBlocks.end());
    assert(blackBlocks.count(incomingPred));
    greyBlocks.clear();
  }
  return std::move(result);
}

//===----------------------------------------------------------------------===//
// OpaqueStorageAllocation: Generate alloc_stack and address projections for all
// abstract storage locations.
//===----------------------------------------------------------------------===//

namespace {
/// Allocate storage on the stack for every opaque value defined in this
/// function in RPO order. If the definition is an argument of this function,
/// simply replace the function argument with an address representing the
/// caller's storage.
///
/// TODO: shrink lifetimes by inserting alloc_stack at the dominance LCA and
/// finding the lifetime boundary with a simple backward walk from uses.
class OpaqueStorageAllocation {
  AddressLoweringState &pass;

public:
  explicit OpaqueStorageAllocation(AddressLoweringState &pass) : pass(pass) {}

  void allocateOpaqueStorage();

protected:
  void convertIndirectFunctionArgs();
  unsigned insertIndirectReturnArgs();
  bool canProjectFrom(std::function<bool(SILValue)> checkDom,
                      SILInstruction *composingUse);
  template <typename C> bool setProjectionFromUser(C innerVals, SILValue value);
  void allocateForBBArg(SILPHIArgument *bbArg);
  void allocateForValue(SILValue value);
  AllocStackInst *createStackAllocation(SILValue value);

  void createStackAllocationStorage(SILValue value) {
    pass.valueStorageMap.getStorage(value).storageAddress =
        createStackAllocation(value);
  }
};
} // end anonymous namespace

/// Top-level entry point: allocate storage for all opaque/resilient values.
void OpaqueStorageAllocation::allocateOpaqueStorage() {
  // Fixup this function's argument types with temporary loads.
  convertIndirectFunctionArgs();

  // Create a new function argument for each indirect result.
  insertIndirectReturnArgs();

  // Populate valueStorageMap.
  OpaqueValueVisitor(pass).mapValueStorage();

  // Create an AllocStack for every opaque value defined in the function.  Visit
  // values in post-order to create storage for aggregates before subobjects.
  // 
  // WARNING: This may split critical edges (in ValueLifetimeAnalysis).
  for (auto &valueStorageI : reversed(pass.valueStorageMap)) {
    SILValue value = valueStorageI.value;
    if (auto *bbArg = dyn_cast<SILPHIArgument>(value))
      allocateForBBArg(bbArg);
    else
      allocateForValue(value);
  }
}

/// Replace each value-typed argument to the current function with an
/// address-typed argument by inserting a temporary load instruction.
void OpaqueStorageAllocation::convertIndirectFunctionArgs() {
  // Insert temporary argument loads at the top of the function.
  SILBuilderWithScope argBuilder(pass.F->getEntryBlock()->begin());
  argBuilder.setSILConventions(
      SILModuleConventions::getLoweredAddressConventions());
  argBuilder.setOpenedArchetypesTracker(&pass.openedArchetypesTracker);

  auto fnConv = pass.F->getConventions();
  unsigned argIdx = fnConv.getSILArgIndexOfFirstParam();
  for (SILParameterInfo param :
       pass.F->getLoweredFunctionType()->getParameters()) {

    if (param.isFormalIndirect() && !fnConv.isSILIndirect(param)) {
      SILArgument *arg = pass.F->getArgument(argIdx);
      SILType addrType = arg->getType().getAddressType();

      LoadInst *loadArg = argBuilder.createLoad(
          SILValue(arg).getLoc(), SILUndef::get(addrType, pass.F->getModule()),
          LoadOwnershipQualifier::Unqualified);

      arg->replaceAllUsesWith(loadArg);
      assert(!pass.valueStorageMap.contains(arg));

      arg = arg->getParent()->replaceFunctionArgument(
          arg->getIndex(), addrType, ValueOwnershipKind::Trivial,
          arg->getDecl());

      loadArg->setOperand(arg);

      if (addrType.isAddressOnly(pass.F->getModule()))
        pass.valueStorageMap.insertValue(loadArg).storageAddress = arg;
    }
    ++argIdx;
  }
  assert(argIdx
         == fnConv.getSILArgIndexOfFirstParam() + fnConv.getNumSILArguments());
}

/// Insert function arguments for any @out result type. Return the number of
/// indirect result arguments added.
unsigned OpaqueStorageAllocation::insertIndirectReturnArgs() {
  auto &ctx = pass.F->getModule().getASTContext();
  unsigned argIdx = 0;
  for (auto resultTy : pass.loweredFnConv.getIndirectSILResultTypes()) {
    auto bodyResultTy = pass.F->mapTypeIntoContext(resultTy);
    auto var = new (ctx)
        ParamDecl(VarDecl::Specifier::InOut, SourceLoc(), SourceLoc(),
                  ctx.getIdentifier("$return_value"), SourceLoc(),
                  ctx.getIdentifier("$return_value"),
                  bodyResultTy.getSwiftRValueType(), pass.F->getDeclContext());

    pass.F->begin()->insertFunctionArgument(argIdx,
                                            bodyResultTy.getAddressType(),
                                            ValueOwnershipKind::Trivial, var);
    ++argIdx;
  }
  assert(argIdx == pass.loweredFnConv.getNumIndirectSILResults());
  return argIdx;
}

/// Is this operand composing an aggregate from a subobject, or simply
/// forwarding the operand's value to storage defined elsewhere?
//
// \param checkDom Function that checks if the specified storage definition
// dominates all necessary uses.
//
// This does not allow projections from terminators. That is handled by
// BlockStorageOptimizer.
bool OpaqueStorageAllocation::canProjectFrom(
    std::function<bool(SILValue)> checkDom, SILInstruction *composingUse) {
  if (!OptimizeOpaqueAddressLowering)
    return false;

  SILValue composingValue;
  switch (composingUse->getKind()) {
  default:
    return false;
  case SILInstructionKind::ApplyInst:
    // @in operands never need their own storage since they are non-mutating
    // uses. They simply reuse the storage allocated for their operand. So it
    // wouldn't make sense to "project" out of the apply argument.
    return false;
  case SILInstructionKind::EnumInst:
    composingValue = cast<EnumInst>(composingUse);
    break;
  case SILInstructionKind::InitExistentialValueInst: {
    // Ensure that all opened archetypes are available at the inner value's
    // definition.
    auto *initExistential = cast<InitExistentialValueInst>(composingUse);
    for (Operand &operand : initExistential->getTypeDependentOperands()) {
      if (!checkDom(operand.get()))
        return false;
    }
    composingValue = initExistential;
    break;
  }
  case SILInstructionKind::ReturnInst:
    return true;
  case SILInstructionKind::StructInst:
    composingValue = cast<StructInst>(composingUse);
    break;
  case SILInstructionKind::TupleInst:
    composingValue = cast<TupleInst>(composingUse);
    break;
  }
  ValueStorage &storage = pass.valueStorageMap.getStorage(composingValue);
  if (SILValue addr = storage.storageAddress) {
    if (auto *stackInst = dyn_cast<AllocStackInst>(addr))
      return checkDom(stackInst);

    if (isa<SILFunctionArgument>(addr)) {
      return true;
    }
  } else if (storage.isUseProjection) {
    SILValue projVal = pass.valueStorageMap.getProjectedStorage(storage).value;
    return canProjectFrom(checkDom, projVal->getDefiningInstruction());
  }
  return false;
}

/// Find a use of this value that can provide storage for this value.
// \param C is a Range of SILValues. e.g. ArrayRef<SILValue>.
// \param storage is the given value's storage.
template <typename C>
bool OpaqueStorageAllocation::setProjectionFromUser(C innerVals,
                                                    SILValue value) {

  auto checkDom = [&](SILValue storageDef) {
    for (SILValue innerVal : innerVals) {
      if (auto *innerInst = innerVal->getDefiningInstruction()) {
        if (!pass.domInfo->properlyDominates(storageDef, innerInst))
          return false;
      } else {
        auto *bbArg = cast<SILPHIArgument>(innerVal);
        // For block arguments, the storage def's block must structly dominate
        // the argument's block.
        if (!pass.domInfo->properlyDominates(innerVal,
                                             &*bbArg->getParent()->begin())) {
          return false;
        }
      }
    }
    return true;
  };

  for (Operand *use : value->getUses()) {
    if (canProjectFrom(checkDom, use->getUser())) {
      DEBUG(llvm::dbgs() << "  PROJECT "; use->getUser()->dump();
            llvm::dbgs() << "  into "; value->dump());
      pass.valueStorageMap.setComposedUseProjection(use);
      return true;
    }
  }
  return false;
}

// Allocate storage for a BB arg. Unlike normal values, this checks all the
// incoming values to determine whether any are also candidates for projection.
void OpaqueStorageAllocation::allocateForBBArg(SILPHIArgument *bbArg) {
  if (auto *predBB = bbArg->getParent()->getSinglePredecessorBlock()) {
    // switch_enum arguments are different than normal phi-like arguments. The
    // incoming value uses its own storage, and the block argument is always a
    // projection of that storage.
    if (auto *SEI = dyn_cast<SwitchEnumInst>(predBB->getTerminator())) {
      Operand *incomingOper = &SEI->getAllOperands()[0];
      assert(incomingOper->get() == bbArg->getSingleIncomingValue());
      pass.valueStorageMap.setExtractedDefOperand(incomingOper);
      return;
    }
    // try_apply is handled differently. If it returns a tuple, then the bbarg
    // has no storage. If it returns a single value then the bb arg has storage,
    // but that storage isn't projected onto any incoming value.
    if (isa<TryApplyInst>(predBB->getTerminator())) {
      // FIXME: multi-result calls.
      if (!bbArg->getType().is<TupleType>()) {
        if (!setProjectionFromUser(ArrayRef<SILValue>(bbArg), bbArg)) {
          createStackAllocationStorage(bbArg);
        }
      }
      return;
    }
  }
  // BlockArgumentStorageOptimizer computes the incoming values of a basic block
  // argument that can share storage with the block argument. The algorithm
  // processes all incoming values at once, so it is is run when visiting the
  // block argument.
  //
  // The incoming value projections are computed first to give them
  // priority. Then we determine if the block argument itself can share storage
  // with one of its users, given that it may already have projections to
  // incoming values.
  //
  // The single-incoming value case (including try_apply results) will be
  // immediately pruned--it will always be a projection of its block argument.
  auto argStorageResult =
      BlockArgumentStorageOptimizer(bbArg).computeArgumentProjections();

  if (!setProjectionFromUser(argStorageResult.getIncomingValueRange(), bbArg)) {
    createStackAllocationStorage(bbArg);
  }

  // Regardless of whether we projected from a user or allocated storage,
  // provide this storage to all the incoming values that can reuse it.
  for (Operand *argOper : argStorageResult.getArgumentProjections())
    pass.valueStorageMap.setComposedUseProjection(argOper);
}

static Operand *getBorrowedStorageOperand(SILValue value) {
  switch (value->getKind()) {
  default:
    return nullptr;

  case ValueKind::TupleExtractInst:
    // TupleExtract from an apply are handled specially until we have
    // multi-result calls. Force them to allocate storage.
    if (auto *TEI = dyn_cast<TupleExtractInst>(value)) {
      if (ApplySite::isa(TEI->getOperand())
          || isa<SILPHIArgument>(TEI->getOperand())) {
        return nullptr;
      }
    }
    LLVM_FALLTHROUGH;
  case ValueKind::StructExtractInst:
  case ValueKind::OpenExistentialValueInst:
  case ValueKind::OpenExistentialBoxValueInst:
    assert(value.getOwnershipKind() == ValueOwnershipKind::Guaranteed);
    return &cast<SingleValueInstruction>(value)->getAllOperands()[0];
  }
}

// Return true if this value can reuse storage projected from one of its uses.
// \param storage is the value's storage information.
static bool canProjectTo(SILValue value, ValueStorage &storage) {
  // Function arguments use caller storage. The proxy load will be mapped to an
  // address-type SILFunctionArgument. Since storage is already mapped in that
  // case, we shouldn't reach here.
  assert(!storage.storageAddress);

  // TODO: Remove this with multi-result calls.
  if (auto apply = ApplySite::isa(value)) {
    // Result tuples will be canonicalized during apply rewriting so the tuple
    // itself is unused.
    if (value->getType().is<TupleType>()) {
      assert(apply.getSubstCalleeType()->getNumResults() > 1);
      return false;
    }
  }

  // Values that borrow their operand inherit storage from that operand, not
  // their use.
  if (getBorrowedStorageOperand(value))
    return false;

  return true;
}

/// Allocate storage for a single opaque/resilient value.
void OpaqueStorageAllocation::allocateForValue(SILValue value) {
  // Function args are not inserted in valuestorageMap. Instead, a
  // proxy load instruction is inserted.
  assert(!isa<SILFunctionArgument>(value));

  auto &storage = pass.valueStorageMap.getStorage(value);

  // Function and argument loads already have a storage address before
  // allocating value storage.
  //
  // Inputs to block arguments may have already projected storage from the block
  // arguments.
  if (storage.isAllocated())
    return;

  // Attempt to reuse a user's storage.
  if (canProjectTo(value, storage)
      && setProjectionFromUser(ArrayRef<SILValue>(value), value)) {
    return;
  }
  // Temporary special case. Result tuples will be canonicalized during apply
  // rewriting so the tuple itself is unused.
  // TODO: multi-result calls.
  if (auto apply = ApplySite::isa(value)) {
    if (value->getType().is<TupleType>()) {
      assert(apply.getSubstCalleeType()->getNumResults() > 1);
      return;
    }
  }
  // Values that borrow their operand inherit storage from that operand.
  if (auto *storageOper = getBorrowedStorageOperand(value)) {
    pass.valueStorageMap.setExtractedDefOperand(storageOper);
    return;
  }

  createStackAllocationStorage(value);
}

// Create alloc_stack and jointly-postdominating dealloc_stack instructions.
// Nesting will be fixed later.
// 
// This may split critical edges. If so, it updates pass.domInfo.
AllocStackInst *OpaqueStorageAllocation::createStackAllocation(SILValue value) {
  SILType allocTy = value->getType();
  auto *defInst = value->getDefiningInstruction();

  auto createAllocStack = [&](SILInstruction *allocPoint,
                              ArrayRef<SILInstruction *> deallocPoints) {
    SILBuilderWithScope allocBuilder(allocPoint);
    allocBuilder.setSILConventions(
        SILModuleConventions::getLoweredAddressConventions());
    allocBuilder.setOpenedArchetypesTracker(&pass.openedArchetypesTracker);
    // FIXME: debug scope of block arguments?
    if (defInst)
      allocBuilder.setCurrentDebugScope(defInst->getDebugScope());

    AllocStackInst *allocInstr =
        allocBuilder.createAllocStack(value.getLoc(), allocTy);

    for (SILInstruction *deallocPoint : deallocPoints) {
      SILBuilderWithScope B(deallocPoint);
      if (defInst)
        B.setCurrentDebugScope(defInst->getDebugScope());
      B.createDeallocStack(value.getLoc(), allocInstr);
    }
    return allocInstr;
  };

  if (allocTy.isOpenedExistential()) {
    // Don't allow non-instructions to have opened archetypes.
    auto *def = cast<SingleValueInstruction>(value);

    // OpenExistential-ish instructions have guaranteed lifetime--they project
    // their operand's storage--so should never reach here.
    assert(!getOpenedArchetypeOf(def));

    // For all other instructions, just allocate storage immediately before the
    // value is defined.
    DeadEndBlocks DEBlocks(pass.F);
    llvm::SmallVector<SILInstruction *, 8> UsePoints;
    ValueLifetimeAnalysis VLA(def);
    ValueLifetimeAnalysis::Frontier Frontier;
    // This updates DominanceInfo if it splits a critical edge.
    VLA.computeFrontier(Frontier, ValueLifetimeAnalysis::AllowToModifyCFG,
                        &DEBlocks, pass.domInfo);
    return createAllocStack(def, Frontier);
  }
  return createAllocStack(&*pass.F->begin()->begin(), pass.exitingInsts);
}

//===----------------------------------------------------------------------===//
// AddressMaterialization - materialize storage addresses, generate projections.
//===----------------------------------------------------------------------===//

namespace {
/// Materialize the address of a value's storage. For values that are directly
/// mapped to a storage location, simply return the mapped `AllocStackInst`.
/// For subobjects emit any necessary `_addr` projections using the provided
/// `SILBuilder`.
///
/// This is a common utility for ApplyRewriter, AddressOnlyDefRewriter,
/// and AddressOnlyUseRewriter.
class AddressMaterialization {
  AddressLoweringState &pass;
  SILBuilder &B;

public:
  AddressMaterialization(AddressLoweringState &pass, SILBuilder &B)
      : pass(pass), B(B) {}

  SILValue initializeOperandMem(Operand *operand);

  SILValue materializeAddress(SILValue origValue);

  SILValue materializeProjectionFromDef(Operand *operand, SILValue origValue);
  SILValue materializeProjectionFromUse(Operand *operand);
};
} // anonymous namespace

/// Given the operand of an aggregate instruction (struct, tuple, enum),
/// materialize an address pointing to memory for this operand and ensure that
/// this memory is initialized with the subobject. Generates the address
/// projection and copy if needed.
SILValue AddressMaterialization::initializeOperandMem(Operand *operand) {
  SILValue def = operand->get();
  SILValue destAddr;
  if (operand->get()->getType().isAddressOnly(pass.F->getModule())) {
    ValueStorage &storage = pass.valueStorageMap.getStorage(def);
    // Source value should already be rewritten.
    assert(storage.isRewritten);
    if (pass.valueStorageMap.isUseProjection(operand))
      destAddr = storage.storageAddress;
    else {
      destAddr = materializeProjectionFromUse(operand);
      B.createCopyAddr(operand->getUser()->getLoc(), storage.storageAddress,
                       destAddr, IsTake, IsInitialization);
    }
  } else {
    destAddr = materializeProjectionFromUse(operand);
    B.createStore(operand->getUser()->getLoc(), operand->get(), destAddr,
                  StoreOwnershipQualifier::Unqualified);
  }
  return destAddr;
}

/// Return the address of the storage for `origValue`. This may involve
/// materializing projections.
///
/// As a side effect, record the materialized address as storage for origValue.
SILValue AddressMaterialization::materializeAddress(SILValue origValue) {
  ValueStorage &storage = pass.valueStorageMap.getStorage(origValue);

  if (storage.storageAddress)
    return storage.storageAddress;

  // Handle a value that is composed by a user (struct/tuple/enum).
  if (storage.isUseProjection) {
    Operand *use = pass.valueStorageMap.getUseProjectionOperand(storage);
    storage.storageAddress = materializeProjectionFromUse(use);
    return storage.storageAddress;
  }
  // Handle a value that is extracted from an aggregate.
  assert(storage.isDefProjection);
  auto *extractInst = cast<SingleValueInstruction>(origValue);
  Operand *defOper =
      &extractInst->getAllOperands()[storage.projectedOperandNum];
  storage.storageAddress = materializeProjectionFromDef(defOper, origValue);
  return storage.storageAddress;
}

/// Materialize the address of a subobject extracted from this operand by this
/// operand's user. origValue is be the value associated with the subobject
/// storage. Normally it will be the operand's user, except when it is a block
/// argument for a switch_enum.
SILValue
AddressMaterialization::materializeProjectionFromDef(Operand *operand,
                                                     SILValue origValue) {
  SILInstruction *user = operand->getUser();
  switch (user->getKind()) {
  default:
    llvm_unreachable("Unexpected projection from def.");
  case SILInstructionKind::StructExtractInst: {
    auto *extractInst = cast<StructExtractInst>(user);
    SILValue srcAddr = materializeAddress(extractInst->getOperand());

    return B.createStructElementAddr(extractInst->getLoc(), srcAddr,
                                     extractInst->getField(),
                                     extractInst->getType().getAddressType());
  }
  case SILInstructionKind::TupleExtractInst: {
    auto *extractInst = cast<TupleExtractInst>(user);
    SILValue srcAddr = materializeAddress(extractInst->getOperand());

    return B.createTupleElementAddr(extractInst->getLoc(), srcAddr,
                                    extractInst->getFieldNo(),
                                    extractInst->getType().getAddressType());
  }
  case SILInstructionKind::SwitchEnumInst: {
    auto *SEI = cast<SwitchEnumInst>(user);
    // SwitchEnum is special because the composed operand isn't actually an
    // operand of origValue, which is itself a block argument.
    auto *destBB = cast<SILPHIArgument>(origValue)->getParent();
    SILValue enumAddr =
        pass.valueStorageMap.getStorage(operand->get()).storageAddress;
    auto eltDecl = SEI->getUniqueCaseForDestination(destBB);
    assert(eltDecl && "No unique case found for destination block");
    return B.createUncheckedTakeEnumDataAddr(SEI->getLoc(), enumAddr,
                                             eltDecl.get());
  }
  case SILInstructionKind::UncheckedEnumDataInst: {
    llvm_unreachable("unchecked_enum_data unimplemented"); //!!!
  }
  }
}

/// Materialize the address of a subobject composed by this operand. The
/// operand's user is an aggregate (struct, tuple, enum,
/// init_existential_value), or a terminator that reuses storage from a block
/// argument.
SILValue
AddressMaterialization::materializeProjectionFromUse(Operand *operand) {
  SILInstruction *user = operand->getUser();

  // Recurse through block arguments.
  if (isa<TermInst>(user))
    return materializeAddress(getValueOfOperandUse(operand));

  switch (user->getKind()) {
  default:
#ifndef NDEBUG
    user->dump();
#endif
    llvm_unreachable("Unexpected projection from use.");
  case SILInstructionKind::EnumInst: {
    auto *enumInst = cast<EnumInst>(user);
    SILValue enumAddr = materializeAddress(enumInst);
    return B.createInitEnumDataAddr(enumInst->getLoc(), enumAddr,
                                    enumInst->getElement(),
                                    operand->get()->getType().getAddressType());
  }
  case SILInstructionKind::InitExistentialValueInst: {
    auto *initExistentialValue = cast<InitExistentialValueInst>(user);
    SILValue containerAddr = materializeAddress(initExistentialValue);
    auto canTy = initExistentialValue->getFormalConcreteType();
    auto opaque = Lowering::AbstractionPattern::getOpaque();
    auto &concreteTL = pass.F->getModule().Types.getTypeLowering(opaque, canTy);
    return B.createInitExistentialAddr(
        initExistentialValue->getLoc(), containerAddr, canTy,
        concreteTL.getLoweredType(), initExistentialValue->getConformances());
  }
  case SILInstructionKind::ReturnInst: {
    assert(pass.loweredFnConv.hasIndirectSILResults());
    return pass.F->getArguments()[0];
  }
  case SILInstructionKind::StructInst: {
    auto *structInst = cast<StructInst>(user);

    auto fieldIter = structInst->getStructDecl()->getStoredProperties().begin();
    std::advance(fieldIter, operand->getOperandNumber());

    SILValue structAddr = materializeAddress(structInst);
    return B.createStructElementAddr(
        structInst->getLoc(), structAddr, *fieldIter,
        operand->get()->getType().getAddressType());
  }
  case SILInstructionKind::TupleInst: {
    auto *tupleInst = cast<TupleInst>(user);
    // Function return values.
    if (tupleInst->hasOneUse()
        && isa<ReturnInst>(tupleInst->use_begin()->getUser())) {
      unsigned resultIdx = tupleInst->getElementIndex(operand);
      assert(resultIdx < pass.loweredFnConv.getNumIndirectSILResults());
      // Cannot call getIndirectSILResults here because that API uses the
      // original function type.
      return pass.F->getArguments()[resultIdx];
    }
    SILValue tupleAddr = materializeAddress(tupleInst);
    return B.createTupleElementAddr(tupleInst->getLoc(), tupleAddr,
                                    operand->getOperandNumber(),
                                    operand->get()->getType().getAddressType());
  }
  }
}

//===----------------------------------------------------------------------===//
// ApplyRewriter - rewrite call sites with indirect arguments.
//===----------------------------------------------------------------------===//

namespace {
/// Rewrite an Apply, lowering its indirect SIL arguments.
///
/// This can be used one parameter at a time, to gradually rewrite the incoming
/// parameters on the use-side from object to address-type arguments. See
/// rewriteIndirectParameter() and rewriteParameters().
///
/// Once any result needs to be rewritten, then entire apply is replaced. See
/// convertApplyWithIndirectResults(). New indirect result arguments for this
/// function to represent the caller's storage.
class ApplyRewriter {
  AddressLoweringState &pass;
  ApplySite apply;
  SILBuilderWithScope argBuilder;
  AddressMaterialization addrMat;

public:
  ApplyRewriter(ApplySite origCall, AddressLoweringState &pass)
    : pass(pass), apply(origCall), argBuilder(origCall.getInstruction()),
      addrMat(pass, argBuilder)
  {
    argBuilder.setSILConventions(
        SILModuleConventions::getLoweredAddressConventions());
  }

  void rewriteParameters();
  void rewriteIndirectParameter(Operand *operand);

  void convertApplyWithIndirectResults();

protected:
  SILBasicBlock::iterator getCallResultInsertionPoint() {
    if (isa<ApplyInst>(apply))
      return std::next(SILBasicBlock::iterator(apply.getInstruction()));

    auto *bbArg = getTryApplyPseudoResult(cast<TryApplyInst>(apply));
    return bbArg->getParent()->begin();
  }

  void canonicalizeResults(MutableArrayRef<SILValue> directResultValues,
                           ArrayRef<Operand *> nonCanonicalUses);
  SILValue materializeIndirectResultAddress(SILValue origDirectResultVal,
                                            SILType argTy);
};
} // end anonymous namespace

/// Rewrite any indirect parameter in place.
void ApplyRewriter::rewriteParameters() {
  // Rewrite all incoming indirect operands.
  unsigned calleeArgIdx = apply.getCalleeArgIndexOfFirstAppliedArg();
  for (Operand &operand : apply.getArgumentOperands()) {
    if (operand.get()->getType().isObject()) {
      auto argConv =
          apply.getSubstCalleeConv().getSILArgumentConvention(calleeArgIdx);
      if (argConv.isIndirectConvention())
        rewriteIndirectParameter(&operand);
    }
    ++calleeArgIdx;
  }
}

static void insertStackDeallocation(AllocStackInst *allocInst,
                                    SILBasicBlock::iterator insertionPoint) {
  SILBuilder deallocBuilder(insertionPoint);
  deallocBuilder.setSILConventions(
      SILModuleConventions::getLoweredAddressConventions());
  deallocBuilder.setCurrentDebugScope(allocInst->getDebugScope());
  deallocBuilder.createDeallocStack(allocInst->getLoc(), allocInst);
}

/// Deallocate temporary call-site stack storage.
///
/// `argLoad` is non-null for @out args that are loaded.
static void insertStackDeallocationAtCall(AllocStackInst *allocInst,
                                          SILInstruction *applyInst,
                                          SILInstruction *argLoad) {
  SILInstruction *lastUse = argLoad ? argLoad : applyInst;

  switch (applyInst->getKind()) {
  case SILInstructionKind::ApplyInst: {
    insertStackDeallocation(allocInst, std::next(lastUse->getIterator()));
    break;
  }
  case SILInstructionKind::TryApplyInst: {
    auto *TAI = cast<TryApplyInst>(applyInst);
    if (&*lastUse == applyInst)
      insertStackDeallocation(allocInst, TAI->getNormalBB()->begin());
    else
      insertStackDeallocation(allocInst, std::next(lastUse->getIterator()));

    insertStackDeallocation(allocInst, TAI->getErrorBB()->begin());
    break;
  }
  case SILInstructionKind::PartialApplyInst:
    llvm_unreachable("partial apply cannot have indirect results.");
  default:
    llvm_unreachable("not implemented for this instruction!");
  }
}

/// Rewrite a formally indirect parameter in place.
/// Update the operand to the incoming value's storage address.
/// After this, the SIL argument types no longer match SIL function conventions.
///
/// Temporary argument storage may be created for loadable values.
///
/// Note: Temporary argument storage does not own its value. If the argument
/// is owned, the stored value should already have been copied.
void ApplyRewriter::rewriteIndirectParameter(Operand *operand) {
  SILValue argValue = operand->get();

  if (argValue->getType().isAddressOnly(pass.F->getModule())) {
    ValueStorage &storage = pass.valueStorageMap.getStorage(argValue);
    // Source value should already be rewritten.
    assert(storage.isRewritten);
    operand->set(storage.storageAddress);
    return;
  }
  // Allocate temporary storage for a loadable operand.
  AllocStackInst *allocInstr =
      argBuilder.createAllocStack(apply.getLoc(), argValue->getType());

  argBuilder.createStore(apply.getLoc(), argValue, allocInstr,
                         StoreOwnershipQualifier::Unqualified);

  operand->set(allocInstr);

  insertStackDeallocationAtCall(allocInstr, apply.getInstruction(),
                                /*argLoad=*/nullptr);
}

// Canonicalize call result uses. Treat each result of a multi-result call as
// an independent value. Currently, SILGen may generate tuple_extract for each
// result but generate a single destroy_value for the entire tuple of
// results. This makes it impossible to reason about each call result as an
// independent value according to the callee's function type.
//
// directResultValues has an entry for each tuple extract corresponding to
// that result if one exists. This function will add an entry to
// directResultValues whenever it needs to materialize a TupleExtractInst.
void ApplyRewriter::canonicalizeResults(
    MutableArrayRef<SILValue> directResultValues,
    ArrayRef<Operand *> nonCanonicalUses) {

  for (Operand *operand : nonCanonicalUses) {
    SILInstruction *releaseInst = dyn_cast<ReleaseValueInst>(operand->getUser());
    assert(releaseInst && "Simultaneous use of multiple call results.");

    for (unsigned resultIdx : indices(directResultValues)) {
      SILValue result = directResultValues[resultIdx];
      if (!result) {
        SILBuilder resultBuilder(getCallResultInsertionPoint());
        resultBuilder.setSILConventions(
            SILModuleConventions::getLoweredAddressConventions());
        resultBuilder.setCurrentDebugScope(
            apply.getInstruction()->getDebugScope());
        result = resultBuilder.createTupleExtract(
            apply.getInstruction()->getLoc(), getCallPseudoResult(apply),
            resultIdx);
        directResultValues[resultIdx] = result;
      }
      SILBuilderWithScope B(releaseInst);
      B.setSILConventions(SILModuleConventions::getLoweredAddressConventions());
      B.emitDestroyValueOperation(releaseInst->getLoc(), result);
    }
    releaseInst->eraseFromParent();
  }
}

/// Return the storage address for the indirect result corresponding to the
/// given original result value. Allocate temporary argument storage for any
/// indirect results that are unmapped because they are loadable or unused.
///
/// origDirectResultVal may be nullptr for unused results.
SILValue
ApplyRewriter::materializeIndirectResultAddress(SILValue origDirectResultVal,
                                                SILType argTy) {

  if (origDirectResultVal
      && origDirectResultVal->getType().isAddressOnly(pass.F->getModule())) {

    // For normal calls, the tuple_extract result may not have been visited yet.
    addrMat.materializeAddress(origDirectResultVal);

    auto &storage = pass.valueStorageMap.getStorage(origDirectResultVal);
    storage.markRewritten();
    return storage.storageAddress;
  }
  // Allocate temporary call-site storage for an unused or loadable result.
  SILInstruction *origCallInst = apply.getInstruction();
  SILLocation loc = origCallInst->getLoc();
  auto *allocInst = argBuilder.createAllocStack(loc, argTy);
  LoadInst *loadInst = nullptr;
  if (origDirectResultVal) {
    // TODO: Find the try_apply's result block.
    // Build results outside-in to next stack allocations.
    SILBuilder resultBuilder(getCallResultInsertionPoint());
    resultBuilder.setSILConventions(
        SILModuleConventions::getLoweredAddressConventions());
    resultBuilder.setCurrentDebugScope(origCallInst->getDebugScope());
    // This is a formally indirect argument, but is loadable.
    loadInst = resultBuilder.createLoad(loc, allocInst,
                                        LoadOwnershipQualifier::Unqualified);
    origDirectResultVal->replaceAllUsesWith(loadInst);
    if (auto *resultInst = origDirectResultVal->getDefiningInstruction())
      pass.markDead(resultInst);
  }
  insertStackDeallocationAtCall(allocInst, origCallInst, loadInst);
  return SILValue(allocInst);
}

/// Allocate storage for formally indirect results at the given call site.
/// Create a new call instruction with indirect SIL arguments.
///
/// TODO: This is kind of a mess.
void ApplyRewriter::convertApplyWithIndirectResults() {
  assert(apply.getSubstCalleeType()->hasIndirectFormalResults());

  auto *origCallInst = apply.getInstruction();
  SILFunctionConventions origFnConv = apply.getSubstCalleeConv();
  // FIXME: multi-result calls.
  SILValue origCallValue = getCallPseudoResult(apply);

  // Gather the original direct return values.
  // Canonicalize results so no user uses more than one result.
  //
  // FIXME!!!: We should probably canonicalize before we begin rewriting.
  SmallVector<SILValue, 8> origDirectResultValues(
      origFnConv.getNumDirectSILResults());
  SmallVector<Operand *, 4> nonCanonicalUses;
  if (origCallValue->getType().is<TupleType>()) {
    for (Operand *operand : origCallValue->getUses()) {
      if (auto *extract = dyn_cast<TupleExtractInst>(operand->getUser()))
        origDirectResultValues[extract->getFieldNo()] = extract;
      else
        nonCanonicalUses.push_back(operand);
    }
    if (!nonCanonicalUses.empty())
      canonicalizeResults(origDirectResultValues, nonCanonicalUses);
  } else {
    // This call has a single result. Convert it to an indirect
    // result. (convertApplyWithIndirectResults is only invoked for calls with
    // at least one indirect result). An unused result can remain
    // unmapped. Temporary storage will be allocated later when fixing up the
    // call's uses.
    assert(origDirectResultValues.size() == 1);
    origDirectResultValues[0] = origCallValue;
  }

  // Prepare to emit a new call instruction.
  SILLocation loc = origCallInst->getLoc();

  // Use this->callBuilder for building incoming arguments and materializing
  // addresses. Use resultBuilder for loading results.
  SILBuilder resultBuilder(getCallResultInsertionPoint());
  resultBuilder.setSILConventions(
      SILModuleConventions::getLoweredAddressConventions());
  resultBuilder.setCurrentDebugScope(origCallInst->getDebugScope());

  // The new call instruction's SIL calling convention.
  SILFunctionConventions loweredCalleeConv(
      apply.getSubstCalleeType(),
      SILModuleConventions::getLoweredAddressConventions());

  // The new call instruction's SIL argument list.
  SmallVector<SILValue, 8> newCallArgs(loweredCalleeConv.getNumSILArguments());

  // Map the original result indices to new result indices.
  SmallVector<unsigned, 8> newDirectResultIndices(
    origFnConv.getNumDirectSILResults());
  // Indices used to populate newDirectResultIndices.
  unsigned oldDirectResultIdx = 0, newDirectResultIdx = 0;

  // The index of the next indirect result argument.
  unsigned newResultArgIdx =
      loweredCalleeConv.getSILArgIndexOfFirstIndirectResult();

  // Visit each result. Redirect results that are now indirect by calling
  // materializeIndirectResultAddress. Results that remain direct will be
  // redirected later. Populate newCallArgs and newDirectResultIndices.
  for_each(
      apply.getSubstCalleeType()->getResults(), origDirectResultValues,
      [&](SILResultInfo resultInfo, SILValue origDirectResultVal) {
        // Assume that all original results are direct in SIL.
        assert(!origFnConv.isSILIndirect(resultInfo));

        if (loweredCalleeConv.isSILIndirect(resultInfo)) {
          SILValue indirectResultAddr = materializeIndirectResultAddress(
              origDirectResultVal, loweredCalleeConv.getSILType(resultInfo));
          // Record the new indirect call argument.
          newCallArgs[newResultArgIdx++] = indirectResultAddr;
          // Leave a placeholder for indirect results.
          newDirectResultIndices[oldDirectResultIdx++] = ~0;
        } else {
          // Record the new direct result, and advance the direct result
          // indices.
          newDirectResultIndices[oldDirectResultIdx++] = newDirectResultIdx++;
        }
        // replaceAllUses will be called later to handle direct results that
        // remain direct results of the new call instruction.
      });

  // Append the existing call arguments to the SIL argument list. They were
  // already lowered to addresses by rewriteIncomingArgument.
  assert(newResultArgIdx == loweredCalleeConv.getSILArgIndexOfFirstParam());
  unsigned origArgIdx = apply.getSubstCalleeConv().getSILArgIndexOfFirstParam();
  for (unsigned endIdx = newCallArgs.size(); newResultArgIdx < endIdx;
       ++newResultArgIdx, ++origArgIdx) {
    newCallArgs[newResultArgIdx] = apply.getArgument(origArgIdx);
  }

  // Collect the original uses before removing a bb arg.
  SmallVector<SILInstruction *, 8> origUsers(getUserRange(origCallValue));

  // Create a new call value to represent the remaining direct uses.
  SILValue newCallValue;
  switch (origCallInst->getKind()) {
  case SILInstructionKind::ApplyInst: {
    auto *AI = cast<ApplyInst>(origCallInst);
    newCallValue = argBuilder.createApply(
        loc, apply.getCallee(), apply.getSubstitutions(), newCallArgs,
        AI->isNonThrowing(), AI->getSpecializationInfo());

    // Update this rewriter's apply, but leave origCallInst around until
    // extracts have been rewritten below. If it's not loadable don't delete it
    // at all until dead code removal because its storage may be tracked.
    this->apply = ApplySite(AI);
    break;
  }
  case SILInstructionKind::TryApplyInst: {
    auto *TAI = cast<TryApplyInst>(origCallInst);
    auto *newCallInst = argBuilder.createTryApply(
        loc, apply.getCallee(), apply.getSubstitutions(), newCallArgs,
        TAI->getNormalBB(), TAI->getErrorBB(), TAI->getSpecializationInfo());

    // Immediately delete the old try_apply (old applies hang around until
    // dead code removal because they define values).
    origCallInst->eraseFromParent();
    origCallInst = nullptr;
    this->apply = ApplySite(newCallInst);

    // Maybe both results are direct.
    if (origCallValue->getType() == loweredCalleeConv.getSILResultType()) {
      newCallValue = origCallValue;
      break;
    }
    auto *resultArg = cast<SILPHIArgument>(origCallValue);

    // Rewriting the apply with a new result type requires erasing any opaque
    // block arguments.  Create dummy loads to stand in for those block
    // arguments until everything has been rewritten. Just load from undef
    // since, for tuple results, there's no storage address to load from.
    LoadInst *loadArg = resultBuilder.createLoad(
        newCallInst->getLoc(),
        SILUndef::get(resultArg->getType().getAddressType(),
                      pass.F->getModule()),
        LoadOwnershipQualifier::Unqualified);

    if (pass.valueStorageMap.contains(resultArg)) {
      assert(!resultArg->getType().is<TupleType>());

      // Storage was materialized by materializeIndirectResultAddress above.
      auto &origStorage = pass.valueStorageMap.getStorage(resultArg);
      assert(origStorage.isRewritten);

      pass.valueStorageMap.replaceValue(resultArg, loadArg);
    }
    resultArg->replaceAllUsesWith(loadArg);
    assert(resultArg->getParent()->getNumArguments() == 1);
    newCallValue = resultArg->getParent()->replacePHIArgument(
        0, loweredCalleeConv.getSILResultType(),
        origCallValue.getOwnershipKind(), resultArg->getDecl());

    // After replacePHIArgument, origCallValue is no more.
    origCallValue = loadArg;
    break;
  }
  case SILInstructionKind::PartialApplyInst:
    llvm_unreachable("partial_apply cannot have indirect results.");
  default:
    llvm_unreachable("unexpected apply kind.");
  }

  // Replace all unmapped uses of the original call with uses of the new call.
  for (SILInstruction *useInst : origUsers) {
    auto *extractInst = dyn_cast<TupleExtractInst>(useInst);
    if (!extractInst) {
      assert(origFnConv.getNumDirectSILResults() == 1);
      assert(pass.valueStorageMap.getStorage(origCallValue).isRewritten);
      continue;
    }
    unsigned origResultIdx = extractInst->getFieldNo();
    auto resultInfo = origFnConv.getResults()[origResultIdx];

    if (extractInst->getType().isAddressOnly(pass.F->getModule())) {
      // Uses of indirect results will be rewritten by AddressOnlyUseRewriter.
      assert(loweredCalleeConv.isSILIndirect(resultInfo));
      // Mark the extract as rewritten now so we don't attempt to convert the
      // call again.
      pass.valueStorageMap.getStorage(extractInst).markRewritten();
      // FIXME: do we need this? It should be placed on the dead list later.
      if (extractInst->use_empty())
        pass.markDead(extractInst);
      continue;
    }
    if (loweredCalleeConv.isSILIndirect(resultInfo)) {
      // This loadable indirect use should already be redirected to a load from
      // the argument storage and marked dead.
      assert(extractInst->use_empty() && pass.isDead(extractInst));
      continue;
    }
    // Either the new call instruction has only a single direct result, or we
    // map the original tuple field to the new tuple field.
    SILValue newResultVal = newCallValue;
    if (loweredCalleeConv.getNumDirectSILResults() > 1) {
      assert(newCallValue->getType().is<TupleType>());
      newResultVal = resultBuilder.createTupleExtract(
          extractInst->getLoc(), newCallValue,
          newDirectResultIndices[origResultIdx]);
    }
    // Since this is a loadable type, there's no associated storage, so
    // erasing the instruction and rewriting is use operands doesn't invalidate
    // any ValueStorage.
    extractInst->replaceAllUsesWith(newResultVal);
    extractInst->eraseFromParent();
  }
  
  // If this call won't be visited as an opaque value def, mark it deleted.
  if (!origCallInst)
    return;

  auto *AI = dyn_cast<ApplyInst>(origCallInst);
  if (!AI)
    return;
  
  if (pass.valueStorageMap.contains(AI))
    return;

  // This call has no storage. If it is used by an address_only tuple_extract,
  // then it will be marked deleted only after that use is marked
  // deleted. Otherwise all its uses must already be dead and it must be marked
  // dead immediately.
  // Note: Simply checking whether this call is address-only is insufficient,
  // because the address-only use may be have been dead to begin with.
  for (Operand *use : AI->getUses()) {
    auto *extract = cast<TupleExtractInst>(use->getUser());
    if (pass.valueStorageMap.contains(extract)) {
    // At least one of the results has storage. The call cannot be marked dead
    // until its results are marked dead.
#ifndef NDEBUG
      pass.callsToDelete.insert(AI);
#endif
      return;
    }
    assert(pass.isDead(extract));
  }
  pass.markDead(AI);
}

//===----------------------------------------------------------------------===//
// ReturnRewriter - rewrite return instructions for indirect results.
//===----------------------------------------------------------------------===//

class ReturnRewriter {
  AddressLoweringState &pass;

public:
  ReturnRewriter(AddressLoweringState &pass) : pass(pass) {}

  void rewriteReturns();

protected:
  void rewriteReturn(ReturnInst *returnInst);
};

void ReturnRewriter::rewriteReturns() {
  for (SILInstruction *termInst : pass.exitingInsts) {
    if (auto *returnInst = dyn_cast<ReturnInst>(termInst))
      rewriteReturn(returnInst);
    else
      assert(isa<ThrowInst>(termInst));
  }
}

void ReturnRewriter::rewriteReturn(ReturnInst *returnInst) {
  auto insertPt = SILBasicBlock::iterator(returnInst);
  auto bbStart = returnInst->getParent()->begin();
  while (insertPt != bbStart) {
    --insertPt;
    if (!isa<DeallocStackInst>(*insertPt))
      break;
  }
  SILBuilderWithScope B(insertPt);
  B.setSILConventions(SILModuleConventions::getLoweredAddressConventions());

  // Gather direct function results.
  unsigned numOrigDirectResults =
      pass.F->getConventions().getNumDirectSILResults();
  SmallVector<SILValue, 8> origDirectResultValues;
  if (numOrigDirectResults == 1)
    origDirectResultValues.push_back(returnInst->getOperand());
  else {
    auto *tupleInst = cast<TupleInst>(returnInst->getOperand());
    origDirectResultValues.append(tupleInst->getElements().begin(),
                                  tupleInst->getElements().end());
    assert(origDirectResultValues.size() == numOrigDirectResults);
  }

  SILFunctionConventions origFnConv(pass.F->getConventions());
  (void)origFnConv;

  // Convert each result.
  SmallVector<SILValue, 8> newDirectResults;
  unsigned newResultArgIdx =
      pass.loweredFnConv.getSILArgIndexOfFirstIndirectResult();

  for_each(
      pass.F->getLoweredFunctionType()->getResults(), origDirectResultValues,
      [&](SILResultInfo resultInfo, SILValue origDirectResultVal) {
        // Assume that all original results are direct in SIL.
        assert(!origFnConv.isSILIndirect(resultInfo));

        if (pass.loweredFnConv.isSILIndirect(resultInfo)) {
          assert(newResultArgIdx
                 < pass.loweredFnConv.getSILArgIndexOfFirstParam());

          SILArgument *resultArg = B.getFunction().getArgument(newResultArgIdx);
          SILType resultTy = origDirectResultVal->getType();
          if (resultTy.isAddressOnly(pass.F->getModule())) {
            ValueStorage &storage =
                pass.valueStorageMap.getStorage(origDirectResultVal);
            assert(storage.isRewritten);
            SILValue resultAddr = storage.storageAddress;
            if (resultAddr != resultArg) {
              // Copy the result from local storage into the result argument.
              B.createCopyAddr(returnInst->getLoc(), resultAddr, resultArg,
                               IsTake, IsInitialization);
            }
          } else {
            // Store the result into the result argument.
            B.createStore(returnInst->getLoc(), origDirectResultVal, resultArg,
                          StoreOwnershipQualifier::Unqualified);
          }
          ++newResultArgIdx;
        } else {
          // Record the direct result for populating the result tuple.
          newDirectResults.push_back(origDirectResultVal);
        }
      });
  assert(newDirectResults.size()
         == pass.loweredFnConv.getNumDirectSILResults());
  SILValue newReturnVal;
  if (newDirectResults.empty()) {
    SILType emptyTy = B.getModule().Types.getLoweredType(
        TupleType::getEmpty(B.getModule().getASTContext()));
    newReturnVal = B.createTuple(returnInst->getLoc(), emptyTy, {});
  } else if (newDirectResults.size() == 1) {
    newReturnVal = newDirectResults[0];
  } else {
    newReturnVal =
        B.createTuple(returnInst->getLoc(),
                      pass.loweredFnConv.getSILResultType(), newDirectResults);
  }
  // Rewrite the returned value.
  SILValue origFullResult = returnInst->getOperand();
  returnInst->setOperand(newReturnVal);
  if (auto *fullResultInst = origFullResult->getDefiningInstruction()) {
    if (!fullResultInst->hasUsesOfAnyResult())
      pass.markDead(fullResultInst);
  }
}

//===----------------------------------------------------------------------===//
// AddressOnlyUseRewriter - rewrite opaque value uses.
//===----------------------------------------------------------------------===//

namespace {
class AddressOnlyUseRewriter
    : SILInstructionVisitor<AddressOnlyUseRewriter> {
  friend SILVisitorBase<AddressOnlyUseRewriter>;
  friend SILInstructionVisitor<AddressOnlyUseRewriter>;

  AddressLoweringState &pass;

  SILBuilder B;
  AddressMaterialization addrMat;

  Operand *currOper;

public:
  explicit AddressOnlyUseRewriter(AddressLoweringState &pass)
      : pass(pass), B(*pass.F), addrMat(pass, B) {
    B.setSILConventions(SILModuleConventions::getLoweredAddressConventions());
  }

  void visitOperand(Operand *operand) {
    currOper = operand;
    // Special handling for opened archetypes because a single result somehow
    // produces both a value of the opened type and the metatype itself :/
    if (operand->getUser()->isTypeDependentOperand(*operand)) {
      CanType openedTy = operand->get()->getType().getSwiftRValueType();
      assert(openedTy->isOpenedExistential());
      SILValue archetypeDef =
          pass.openedArchetypesTracker.getOpenedArchetypeDef(
              CanArchetypeType(openedTy->castTo<ArchetypeType>()));
      operand->set(archetypeDef);
      return;
    }
    visit(operand->getUser());
  }

protected:
  void markRewritten(SILValue oldValue, SILValue addr) {
    auto &storage = pass.valueStorageMap.getStorage(oldValue);
    storage.storageAddress = addr;
    storage.markRewritten();
  }

  void beforeVisit(SILInstruction *I) {
    DEBUG(llvm::dbgs() << "  REWRITE USE "; I->dump());

    B.setInsertionPoint(I);
    B.setCurrentDebugScope(I->getDebugScope());
  }

  void visitSILInstruction(SILInstruction *I) {
#ifndef NDEBUG
    I->dump();
#endif
    llvm_unreachable("Unimplemented opaque use.");
  }

  // Opaque call argument.
  void visitApplyInst(ApplyInst *applyInst) {
    ApplyRewriter(applyInst, pass).rewriteIndirectParameter(currOper);
  }

  // Opaque branch argument.
  void visitBranchInst(BranchInst *BI) {
    SILValue predStorageAddress =
        pass.valueStorageMap.getStorage(currOper->get()).storageAddress;

    unsigned argIdx = BI->getArgIndexOfOperand(currOper->getOperandNumber());
    auto *bbArg = cast<SILPHIArgument>(BI->getDestBB()->getArgument(argIdx));
    SILValue succStorageAddress = addrMat.materializeAddress(bbArg);

    // If this branch argument is not a projection of the block argument, then
    // copy its value in to block arg storage.
    if (predStorageAddress != succStorageAddress) {
      B.createCopyAddr(BI->getLoc(), predStorageAddress, succStorageAddress,
                       IsTake, IsInitialization);
    }
    // Set this operand to Undef. Dead block arguments are removed after
    // rewriting.
    currOper->set(
        SILUndef::get(currOper->get()->getType(), pass.F->getModule()));
  }

  // Opaque checked cast source.
  void visitCheckedCastValueBranchInst(
      CheckedCastValueBranchInst *checkedBranchInst) {
    llvm::report_fatal_error("Unimplemented CheckCastValueBranch use.");
  }

  // Copy from an opaque source operand.
  void visitCopyValueInst(CopyValueInst *copyInst) {
    llvm_unreachable("Unexpected copy_value");
  }

  // Opaque conditional branch argument.
  void visitCondBranchInst(CondBranchInst *CBI) {
    SILValue predStorageAddress =
        pass.valueStorageMap.getStorage(currOper->get()).storageAddress;

    unsigned operIdx = currOper->getOperandNumber();
    bool isTruePath = CBI->isTrueOperandIndex(operIdx);
    unsigned argIdx =
        operIdx
        - (isTruePath ? CBI->getFalseOperands()[0].getOperandNumber()
                      : CBI->getTrueOperands()[0].getOperandNumber());

    auto *destBB = isTruePath ? CBI->getTrueBB() : CBI->getFalseBB();
    auto *bbArg = cast<SILPHIArgument>(destBB->getArgument(argIdx));
    SILValue succStorageAddress = addrMat.materializeAddress(bbArg);

    // If this branch argument is not a projection of the block argument, then
    // copy its value in to block arg storage.
    if (predStorageAddress != succStorageAddress) {
      B.createCopyAddr(CBI->getLoc(), predStorageAddress, succStorageAddress,
                       IsTake, IsInitialization);
    }
    // Set this operand to Undef. Dead block arguments are removed after
    // rewriting.
    //
    // FIXME: Make sure this is done such that ValueStorage projections can't be
    // referenced once we replace an operand with Undef or erase an instruction
    // or argument.
    currOper->set(
        SILUndef::get(currOper->get()->getType(), pass.F->getModule()));

    ValueStorage &storage = pass.valueStorageMap.getStorage(currOper->get());
    currOper->set(storage.storageAddress);
    llvm::report_fatal_error("Untested."); //!!!
  }

  void visitDebugValueInst(DebugValueInst *debugInst) {
    SILValue srcVal = debugInst->getOperand();
    SILValue srcAddr = pass.valueStorageMap.getStorage(srcVal).storageAddress;
    B.createDebugValueAddr(debugInst->getLoc(), srcAddr);
    pass.markDead(debugInst);
  }

  void visitDeinitExistentialValueInst(
      DeinitExistentialValueInst *deinitExistential) {
    llvm::report_fatal_error("Unimplemented DeinitExsitentialValue use.");
  }

  void visitDestroyValueInst(DestroyValueInst *destroyInst) {
    llvm_unreachable("Unexpected destroy_value");
  }

  // Opaque enum payload. Handle EnumInst on the def side to handle both opaque
  // and loadable operands.
  void visitEnumInst(EnumInst *enumInst) {}

  // Initialize an existential with an opaque payload.
  // (Handle InitExistentialValue on the def side to handle both opaque
  // and loadable operands.)
  void
  visitInitExistentialValueInst(InitExistentialValueInst *initExistential) {}

  // Opening an opaque existential. Rewrite the opened existentials here on the
  // use-side because it may produce either loadable or address-only types.
  void
  visitOpenExistentialValueInst(OpenExistentialValueInst *openExistential) {
    assert(currOper
           == pass.valueStorageMap.getDefProjectionOperand(openExistential));
    SILValue srcAddr =
        pass.valueStorageMap.getStorage(currOper->get()).storageAddress;
    // Mutable access is always by address.
    auto *OEA =
        B.createOpenExistentialAddr(openExistential->getLoc(), srcAddr,
                                    openExistential->getType().getAddressType(),
                                    OpenedExistentialAccess::Immutable);
    // Henceforth track this operned archetype using open_existential_addr.
    pass.openedArchetypesTracker.unregisterOpenedArchetypes(openExistential);
    pass.openedArchetypesTracker.registerOpenedArchetypes(OEA);
    markRewritten(openExistential, OEA);
  }

  void visitOpenExistentialBoxValueInst(
      OpenExistentialBoxValueInst *openExistentialBox) {
    llvm::report_fatal_error("Unimplemented OpenExistentialBox use.");
  }

  void visitReleaseValueInst(ReleaseValueInst *releaseInst) {
    SILValue srcVal = releaseInst->getOperand();
    ValueStorage &storage = pass.valueStorageMap.getStorage(srcVal);
    SILValue srcAddr = storage.storageAddress;

    // FIXME: We lower retain_value to retain_value_addr, but lower
    // release_value to destroy_addr. This is confusingly assymetric.  We should
    // remove the release_value_addr instruction complete and just add a flag to
    // destroy_addr to indicate whether IRGen should "outline" its codegen.
    // FIXME: destroy_addr ignores the "atomicity" flag. Do we care?
    B.createDestroyAddr(releaseInst->getLoc(), srcAddr);
    pass.markDead(releaseInst);
  }

  void visitRetainValueInst(RetainValueInst *retainInst) {
    SILValue srcVal = retainInst->getOperand();
    ValueStorage &storage = pass.valueStorageMap.getStorage(srcVal);
    SILValue srcAddr = storage.storageAddress;

    B.createRetainValueAddr(retainInst->getLoc(), srcAddr,
                            retainInst->getAtomicity());
    pass.markDead(retainInst);
  }

  void visitReturnInst(ReturnInst *returnInst) {
    // Returns are rewritten for any function with indirect results after opaque
    // value rewriting.
  }

  void visitSelectValueInst(SelectValueInst *selectInst) {
    llvm::report_fatal_error("Unimplemented SelectValue use.");
  }

  // Opaque enum operand to a switch_enum.
  void visitSwitchEnumInst(SwitchEnumInst *SEI) {
    // The switch_enum should be rewritten to a switch_enum_addr here, but
    // because it's a terminator, it cannot be rewritten until its block
    // arguments are removed.
  }

  void visitStoreInst(StoreInst *storeInst) {
    SILValue srcVal = storeInst->getSrc();
    assert(currOper->get() == srcVal);

    ValueStorage &storage = pass.valueStorageMap.getStorage(srcVal);
    SILValue srcAddr = storage.storageAddress;

    IsTake_t isTakeFlag = IsTake;
    assert(storeInst->getOwnershipQualifier()
           == StoreOwnershipQualifier::Unqualified);

    // Bitwise copy the value. Two locations now share ownership. This is
    // modeled as a take-init.
    B.createCopyAddr(storeInst->getLoc(), srcAddr, storeInst->getDest(),
                     isTakeFlag, IsInitialization);
    pass.markDead(storeInst);
  }

  // Extract from an opaque struct.
  void visitStructExtractInst(StructExtractInst *extractInst) {
    SILValue extractAddr = addrMat.materializeProjectionFromDef(
        &extractInst->getOperandRef(), extractInst);
    if (extractInst->getType().isAddressOnly(pass.F->getModule())) {
      assert(currOper
             == pass.valueStorageMap.getDefProjectionOperand(extractInst));
      markRewritten(extractInst, extractAddr);
    } else {
      assert(!pass.valueStorageMap.contains(extractInst));
      LoadInst *loadElement = B.createLoad(extractInst->getLoc(), extractAddr,
                                           LoadOwnershipQualifier::Unqualified);
      extractInst->replaceAllUsesWith(loadElement);
      pass.markDead(extractInst);
    }
  }

  // Opaque struct member.
  void visitStructInst(StructInst *structInst) {
    // Structs are rewritten on the def-side, where both direct and indirect
    // elements are composed.
  }

  // Opaque call argument.
  void visitTryApplyInst(TryApplyInst *tryApplyInst) {
    ApplyRewriter(tryApplyInst, pass).rewriteIndirectParameter(currOper);
  }

  // Opaque tuple element.
  void visitTupleInst(TupleInst *tupleInst) {
    // Tuples are rewritten on the def-side, where both direct and indirect
    // elements are composed.
  }

  // Extract from an opaque tuple.
  void visitTupleExtractInst(TupleExtractInst *extractInst) {
    // Apply results are rewritten when the result definition is visited.
    if (ApplySite::isa(currOper->get()))
      return;

    SILValue extractAddr = addrMat.materializeProjectionFromDef(
        &extractInst->getOperandRef(), extractInst);
    if (extractInst->getType().isAddressOnly(pass.F->getModule())) {
      assert(currOper
             == pass.valueStorageMap.getDefProjectionOperand(extractInst));
      markRewritten(extractInst, extractAddr);
    } else {
      assert(!pass.valueStorageMap.contains(extractInst));
      LoadInst *loadElement = B.createLoad(extractInst->getLoc(), extractAddr,
                                           LoadOwnershipQualifier::Unqualified);
      extractInst->replaceAllUsesWith(loadElement);
      pass.markDead(extractInst);
    }
    llvm_unreachable("Untested."); //!!!
  }

  void visitUncheckedBitwiseCast(UncheckedBitwiseCastInst *uncheckedCastInst) {
    llvm_unreachable("Unimplemented UncheckedBitwiseCast use.");
  }

  void visitUnconditionalCheckedCastValueInst(
      UnconditionalCheckedCastValueInst *checkedCastInst) {

    llvm_unreachable("Unimplemented UnconditionalCheckedCast use.");
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// AddressOnlyDefRewriter - rewrite opaque value definitions.
//===----------------------------------------------------------------------===//

namespace {
class AddressOnlyDefRewriter
    : SILInstructionVisitor<AddressOnlyDefRewriter> {
  friend SILVisitorBase<AddressOnlyDefRewriter>;
  friend SILInstructionVisitor<AddressOnlyDefRewriter>;

  AddressLoweringState &pass;

  SILBuilder B;
  AddressMaterialization addrMat;

  ValueStorage *storage = nullptr;

public:
  explicit AddressOnlyDefRewriter(AddressLoweringState &pass)
      : pass(pass), B(*pass.F), addrMat(pass, B) {
    B.setSILConventions(SILModuleConventions::getLoweredAddressConventions());
  }

  void visitInst(SILInstruction *inst) { visit(inst); }

  // Set the storage address for am opaque block arg and mark it rewritten.
  // Opaque block args are only removed after all instructions are rewritten.
  void rewriteBBArg(SILPHIArgument *bbArg) {
    B.setInsertionPoint(bbArg->getParent()->begin());
    // TODO: No debug scope for arguments?

    auto &storage = pass.valueStorageMap.getStorage(bbArg);
    assert(!storage.isRewritten);
    storage.storageAddress = addrMat.materializeAddress(bbArg);
    storage.markRewritten();
  }

protected:
  void beforeVisit(SILInstruction *I) {
    // This cast succeeds beecause only specific instructions get added to
    // the value storage map.
    storage = &pass.valueStorageMap.getStorage(cast<SingleValueInstruction>(I));

    DEBUG(llvm::dbgs() << "REWRITE DEF "; I->dump());
    if (storage->storageAddress)
      DEBUG(llvm::dbgs() << "  STORAGE "; storage->storageAddress->dump());

    B.setInsertionPoint(I);
    B.setCurrentDebugScope(I->getDebugScope());
  }

  void visitSILInstruction(SILInstruction *I) {
#ifndef NDEBUG
    I->dump();
#endif
    llvm_unreachable("Unimplemented opaque def.");
  }

  // Call returning a single opaque value.
  void visitApplyInst(ApplyInst *applyInst) {
    assert(isa<SingleValueInstruction>(applyInst) &&
           "beforeVisit assumes that ApplyInst is an SVI");
    assert(!storage->isRewritten);
    // Completely rewrite the apply instruction, handling any remaining
    // (loadable) indirect parameters, allocating memory for indirect
    // results, and generating a new apply instruction.
    ApplyRewriter rewriter(applyInst, pass);
    rewriter.rewriteParameters();
    rewriter.convertApplyWithIndirectResults();
  }

  // Copy into an opaque value.
  void visitCopyValueInst(CopyValueInst *copyInst) {
    assert(storage->isRewritten);
  }

  // Define an opaque enum value.
  void visitEnumInst(EnumInst *enumInst) {
    SILValue enumAddr;
    if (enumInst->hasOperand()) {
      addrMat.initializeOperandMem(&enumInst->getOperandRef());

      assert(storage->storageAddress);
      enumAddr = storage->storageAddress;
    } else
      enumAddr = addrMat.materializeAddress(enumInst);

    B.createInjectEnumAddr(enumInst->getLoc(), enumAddr,
                           enumInst->getElement());

    storage->markRewritten();
  }

  // Define an existential.
  void visitInitExistentialValueInst(
      InitExistentialValueInst *initExistentialValue) {

    // Initialize memory for the operand which may be opaque or loadable.
    addrMat.initializeOperandMem(&initExistentialValue->getOperandRef());

    assert(storage->storageAddress);
    storage->markRewritten();
  }

  // Project an opaque value out of an existential.
  void visitOpenExistentialValueInst(
      OpenExistentialValueInst *openExistentialValue) {
    // Rewritten on the use-side because storage is inherited from the source.
    assert(storage->isRewritten);
  }

  // Project an opaque value out of a box-type existential.
  void visitOpenExistentialBoxValueInst(
      OpenExistentialBoxValueInst *openExistentialBox) {
    llvm::report_fatal_error("Unimplemented OpenExistentialBoxValue def.");
  }

  // Load an opaque value.
  void visitLoadInst(LoadInst *loadInst) {
    // Bitwise copy the value. Two locations now share ownership. This is
    // modeled as a take-init.
    SILValue addr = pass.valueStorageMap.getStorage(loadInst).storageAddress;
    if (addr != loadInst->getOperand()) {
      B.createCopyAddr(loadInst->getLoc(), loadInst->getOperand(), addr, IsTake,
                       IsInitialization);
    }
    storage->markRewritten();
  }

  // Define an opaque struct.
  void visitStructInst(StructInst *structInst) {
    ValueStorage &storage = pass.valueStorageMap.getStorage(structInst);

    // For each element, initialize the operand's memory. Some struct elements
    // may be loadable types.
    for (Operand &operand : structInst->getAllOperands())
      addrMat.initializeOperandMem(&operand);

    storage.markRewritten();
  }

  // Define an opaque tuple.
  void visitTupleInst(TupleInst *tupleInst) {
    ValueStorage &storage = pass.valueStorageMap.getStorage(tupleInst);
    if (storage.isUseProjection
        && isa<ReturnInst>(
               pass.valueStorageMap.getUseProjectionOperand(storage)
                   ->getUser())) {
      // For indirectly returned values, each element has its own storage.
      return;
    }
    // For each element, initialize the operand's memory. Some tuple elements
    // may be loadable types.
    for (Operand &operand : tupleInst->getAllOperands())
      addrMat.initializeOperandMem(&operand);

    storage.markRewritten();
  }

  // Extract an opaque struct member.
  void visitStructExtractInst(StructExtractInst *extractInst) {
    assert(storage->isRewritten);
  }

  // Extract an opaque tuple element.
  void visitTupleExtractInst(TupleExtractInst *extractInst) {
    // If the source is an opaque tuple, as opposed to a call result, then the
    // extract is rewritten on the use-side.
    if (storage->isRewritten)
      return;

    // This must be an indirect result for an apply that has not yet been
    // rewritten. Rewrite the apply.
    SILValue srcVal = extractInst->getOperand();
    if (auto *AI = dyn_cast<ApplyInst>(srcVal)) {
      // Go ahead and rewrite the apply that produces the tuple now since it was
      // effectively just skipped by the def rewriter.
      ApplyRewriter(AI, pass).convertApplyWithIndirectResults();
      assert(storage->isRewritten);
      return;
    }
    assert(isa<SILPHIArgument>(srcVal));
    // Handle multi-result try_apply. Rewriting the apply itself is deferred
    // since it involves a terminator and bb arg.
    storage->storageAddress = addrMat.materializeAddress(extractInst);
    storage->markRewritten();
  }
};
} // end anonymous namespace

// Rewrite applies with indirect paramters or results of loadable types which
// were not visited during opaque value rewritting.
static void rewriteIndirectApply(ApplySite apply, AddressLoweringState &pass) {
  // Some normal calls with indirect formal results have already been rewritten.
  if (apply.getSubstCalleeType()->hasIndirectFormalResults()) {
    bool isRewritten = false;
    if (isa<ApplyInst>(apply)) {
      visitCallResults(apply, [&](SILValue result) {
        if (result->getType().isAddressOnly(pass.F->getModule())) {
          assert(pass.valueStorageMap.getStorage(result).isRewritten);
          isRewritten = true;
          return false;
        }
        return true;
      });
    }
    if (!isRewritten) {
      ApplyRewriter rewriter(apply, pass);
      rewriter.rewriteParameters();
      rewriter.convertApplyWithIndirectResults();
      return;
    }
  }
  ApplyRewriter(apply, pass).rewriteParameters();
}

// Rewrite switch_enum to switch_enum_addr. This can only happen after all
// associated block arguments are removed.
//
// FIXME: Make sure this is done such that ValueStorage projections can't be
// referenced once we replace an operand with Undef or erase an instruction or
// argument.
static void rewriteSwitchEnum(SwitchEnumInst *SEI, AddressLoweringState &pass) {
  SILValue enumVal = SEI->getOperand();
  auto &storage = pass.valueStorageMap.getStorage(enumVal);
  assert(storage.isRewritten);
  SILValue enumAddr = storage.storageAddress;

  // TODO: We should be able to avoid locally copying the case pointers here.
  SmallVector<std::pair<EnumElementDecl *, SILBasicBlock *>, 8> cases;
  SmallVector<ProfileCounter, 8> caseCounters;

  // Collect switch cases for rewriting and remove block arguments.
  for (unsigned caseIdx : range(SEI->getNumCases())) {
    EnumElementDecl *caseDecl;
    SILBasicBlock *caseBB;
    auto caseRecord = SEI->getCase(caseIdx);
    std::tie(caseDecl, caseBB) = caseRecord;

    cases.push_back(caseRecord);
    caseCounters.push_back(SEI->getCaseCount(caseIdx));

    if (caseBB->getArguments().size() == 0)
      continue;

    if (!caseDecl->hasAssociatedValues())
      continue;

    assert(caseBB->getPHIArguments().size() == 1);
    SILPHIArgument *caseArg = caseBB->getPHIArguments()[0];

    SILBuilderWithScope argBuilder(caseBB->begin());
    argBuilder.setSILConventions(
        SILModuleConventions::getLoweredAddressConventions());
    AddressMaterialization addrMat(pass, argBuilder);

    if (caseArg->getType().isAddressOnly(pass.F->getModule())) {
      // Replace the block arg with a dummy because. As a rule, don't delete
      // anything that may define an opaque value until dead code
      // elimination. Also, conceivably we could have a cycle of switch_enum
      // during rewriting.
      LoadInst *loadArg = argBuilder.createLoad(
          SEI->getLoc(),
          SILUndef::get(caseArg->getType().getAddressType(),
                        pass.F->getModule()),
          LoadOwnershipQualifier::Unqualified);

      caseArg->replaceAllUsesWith(loadArg);

      auto &origStorage = pass.valueStorageMap.getStorage(caseArg);
      assert(origStorage.isRewritten);

      pass.valueStorageMap.replaceValue(caseArg, loadArg);

    } else {
      SILValue eltAddr = addrMat.materializeProjectionFromDef(
          &SEI->getAllOperands()[0], caseArg);

      // Rewrite any non-opaque cases now since we don't visit their defs.
      auto *loadElt = argBuilder.createLoad(
          SEI->getLoc(), eltAddr, LoadOwnershipQualifier::Unqualified);

      caseArg->replaceAllUsesWith(loadElt);
    }
    caseBB->eraseArgument(0);
  }
  auto *defaultBB = SEI->hasDefault() ? SEI->getDefaultBB() : nullptr;
  auto defaultCounter =
      SEI->hasDefault() ? SEI->getDefaultCount() : ProfileCounter();

  SILBuilderWithScope B(SEI);
  B.setSILConventions(SILModuleConventions::getLoweredAddressConventions());
  B.createSwitchEnumAddr(SEI->getLoc(), enumAddr, defaultBB, cases,
                         ArrayRef<ProfileCounter>(caseCounters),
                         defaultCounter);
  SEI->eraseFromParent();

  if (auto *unusedI = enumVal->getDefiningInstruction())
    pass.markDead(unusedI);
}

static void rewriteFunction(AddressLoweringState &pass) {
  AddressOnlyDefRewriter defVisitor(pass);
  AddressOnlyUseRewriter useVisitor(pass);

  // For each opaque value, rewrite its users and its defining instruction.
  for (auto &valueStorageI : pass.valueStorageMap) {
    SILValue valueDef = valueStorageI.value;

    // TODO: MultiValueInstruction: ApplyInst
    if (auto *inst = dyn_cast<SingleValueInstruction>(valueDef))
      defVisitor.visitInst(inst);
    else
      defVisitor.rewriteBBArg(cast<SILPHIArgument>(valueDef));

    SmallVector<Operand *, 8> uses(valueDef->getUses());
    for (Operand *oper : uses)
      useVisitor.visitOperand(oper);
  }
  // Rewrite any remaining (loadable) indirect parameters or call results that
  // need to be adjusted for the calling convention change.
  //
  // Also rewrite any try_apply with an indirect result and remove the
  // corresponding block argument.
  //
  // FIXME: remove an apply from this set when it's rewritten.
  for (ApplySite apply : pass.indirectApplies)
    rewriteIndirectApply(apply, pass);

  // Rewrite terminators now that all opaque values are rewritten. Do this after
  // rewriting all opaque value uses so that block arguments can be removed.
  //
  // Any try_apply with indirect parameters or results is already rewritten by
  // rewriteIndirectApply.
  for (auto &bb : *pass.F) {
    if (auto *SEI = dyn_cast<SwitchEnumInst>(bb.getTerminator())) {
      if (SEI->getOperand()->getType().isAddressOnly(pass.F->getModule()))
        rewriteSwitchEnum(SEI, pass);
    }
  }

  // Rewrite this function's return value now that all opaque values within the
  // function are rewritten. This still depends on a valid ValueStorage
  // projection operands.
  if (pass.F->getLoweredFunctionType()->hasIndirectFormalResults())
    ReturnRewriter(pass).rewriteReturns();
}

// Given an array of terminator operand values, produce an array of
// operands with those corresponding to deadArgIndices stripped out.
static void filterDeadArgs(OperandValueArrayRef origArgs,
                           ArrayRef<unsigned> deadArgIndices,
                           SmallVectorImpl<SILValue> &newArgs) {
  auto nextDeadArgI = deadArgIndices.begin();
  for (unsigned i : indices(origArgs)) {
    if (i == *nextDeadArgI) {
      ++nextDeadArgI;
      continue;
    }
    newArgs.push_back(origArgs[i]);
  }
  assert(nextDeadArgI == deadArgIndices.end());
}

// Rewrite a BranchInst omitting dead arguments.
static void removeBranchArgs(BranchInst *BI,
                             SmallVectorImpl<unsigned> &deadArgIndices,
                             AddressLoweringState &pass) {

  llvm::SmallVector<SILValue, 4> branchArgs;
  filterDeadArgs(BI->getArgs(), deadArgIndices, branchArgs);

  SILBuilderWithScope B(BI);
  B.setSILConventions(SILModuleConventions::getLoweredAddressConventions());
  B.createBranch(BI->getLoc(), BI->getDestBB(), branchArgs);

  BI->eraseFromParent();
}

// Rewrite a CondBranchInst omitting dead arguments.
static void removeCondBranchArgs(CondBranchInst *CBI, SILBasicBlock *targetBB,
                                 ArrayRef<unsigned> deadArgIndices,
                                 AddressLoweringState &pass) {
  SmallVector<SILValue, 8> trueArgs;
  SmallVector<SILValue, 8> falseArgs;

  if (targetBB == CBI->getTrueBB())
    filterDeadArgs(CBI->getTrueArgs(), deadArgIndices, trueArgs);
  else
    trueArgs.append(CBI->getTrueArgs().begin(), CBI->getTrueArgs().end());

  if (targetBB == CBI->getFalseBB())
    filterDeadArgs(CBI->getFalseArgs(), deadArgIndices, falseArgs);
  else
    falseArgs.append(CBI->getFalseArgs().begin(), CBI->getFalseArgs().end());

  SILBuilderWithScope B(CBI);
  B.setSILConventions(SILModuleConventions::getLoweredAddressConventions());
  B.createCondBranch(CBI->getLoc(), CBI->getCondition(), CBI->getTrueBB(),
                     trueArgs, CBI->getFalseBB(), falseArgs,
                     CBI->getTrueBBCount(), CBI->getFalseBBCount());
  CBI->eraseFromParent();

  llvm_unreachable("Untested"); //!!!
}

// Remove all opaque block arguments. Their inputs have already been substituted
// with Undef.
//
// FIXME: Generalize to other terminators.
static void removeOpaqueBBArgs(SILBasicBlock *bb, AddressLoweringState &pass) {
  if (bb->isEntry())
    return;

  SmallVector<unsigned, 16> deadArgIndices;
  for (auto *bbArg : bb->getArguments()) {
    if (bbArg->getType().isAddressOnly(pass.F->getModule()))
      deadArgIndices.push_back(bbArg->getIndex());
  }
  if (deadArgIndices.empty())
    return;

  // Iterate while modifying the predecessor's terminators.
  for (auto predI = bb->pred_begin(), nextI = predI, predE = bb->pred_end();
       predI != predE; predI = nextI) {
    ++nextI;
    auto *predTerm = (*predI)->getTerminator();
    switch (predTerm->getTermKind()) {
    default:
      llvm_unreachable("Unexpected block terminator.");

    case TermKind::BranchInst:
      removeBranchArgs(cast<BranchInst>(predTerm), deadArgIndices, pass);
      break;
    case TermKind::CondBranchInst:
      removeCondBranchArgs(cast<CondBranchInst>(predTerm), bb, deadArgIndices,
                           pass);
      break;
    case TermKind::SwitchEnumInst:
      llvm_unreachable("switch_enum arguments are removed when the terminator "
                       "is rewritten.");
      break;
    }
  }
  for (unsigned deadArgIdx : deadArgIndices)
    bb->eraseArgument(deadArgIdx);
}

// pass.instsToDelete now contains instructions that were explicitly marked
// dead. These already have no users. The rest of the address-only definitions
// will be removed bottom-up by visiting valuestorageMap.
//
// Calls are tricky because sometimes they are values, and sometimes they have
// tuple_extracts representing their values. Either way, they still need to be
// deleted in the correct use-def order.
//
// Address-only block arguments that are tied to terminators (switch_enum,
// try_apply) have already been removed and replace with fake load. The phi-like
// block arguments are removed here after all other instructions.
static void deleteRewrittenInstructions(AddressLoweringState &pass) {
  // Add the rest of the instructions to the dead list in post order.
  // FIXME: make sure we cleaned up address-only BB arguments.
  for (auto &valueStorageI : reversed(pass.valueStorageMap)) {
    SILValue val = valueStorageI.value;
    // If the storage was explicitly erased, skip it. (e.g. replaced bb args).
    if (!pass.valueStorageMap.contains(val))
      continue;

#ifndef NDEBUG
    auto &storage = pass.valueStorageMap.getStorage(val);
    // Returned tuples and multi-result calls are currently values without
    // storage.  Everything else must have been rewritten.
    assert(storage.isRewritten
           || (!storage.storageAddress && (isa<TupleInst>(val))
               || ApplySite::isa(val)));
#endif
    // TODO: MultiValueInstruction: ApplyInst
    auto *deadInst = dyn_cast<SingleValueInstruction>(val);
    if (!deadInst) {
      // Just skip block args. Remove all of a block's dead args in one pass to
      // avoid rewriting all predecessors terminators multiple times :(.
      assert(isa<SILPHIArgument>(val));
      continue;
    }

    DEBUG(llvm::dbgs() << "DEAD "; deadInst->dump());
#ifndef NDEBUG
    for (auto result : deadInst->getResults())
      for (Operand *operand : result->getUses())
        assert(pass.instsToDelete.count(operand->getUser()));
#endif
    pass.instsToDelete.insert(deadInst);

    // FIXME: MultiValue. Insert non-value calls into the dead instruction list
    // if all of their results have been deleted.
    auto *extract = dyn_cast<TupleExtractInst>(deadInst);
    if (extract) {
      auto *applyInst = dyn_cast<ApplyInst>(extract->getOperand());
      if (applyInst) {
        if (all_of(applyInst->getUses(),
                   [&](Operand *op) -> bool {
                     return pass.instsToDelete.count(op->getUser());
                   })) {
          DEBUG(llvm::dbgs() << "DEAD "; applyInst->dump());
          pass.instsToDelete.insert(applyInst);
          assert(pass.callsToDelete.remove(applyInst));
        }
      }
    }
  }
  assert(pass.callsToDelete.empty());
  
  pass.valueStorageMap.clear();

  // Delete instructions in postorder
  recursivelyDeleteTriviallyDeadInstructions(pass.instsToDelete.takeVector(),
                                             true);

  // Remove block args after removing all instructions that may use them.
  for (auto &bb : *pass.F)
    removeOpaqueBBArgs(&bb, pass);
}

//===----------------------------------------------------------------------===//
// AddressLowering: Top-Level Module Transform.
//===----------------------------------------------------------------------===//

namespace {
// Note: the only reason this is not a FunctionTransform is to change the SIL
// stage for all functions at once.
class AddressLowering : public SILModuleTransform {
  /// The entry point to this function transformation.
  void run() override;

  void runOnFunction(SILFunction *F);
};
} // end anonymous namespace

void AddressLowering::runOnFunction(SILFunction *F) {
  if (!F->isDefinition())
    return;

  PrettyStackTraceSILFunction FuncScope("address-lowering", F);

  DEBUG(llvm::dbgs() << "Address Lowering: " << F->getName() << "\n");

  auto *DA = PM->getAnalysis<DominanceAnalysis>();

  AddressLoweringState pass(F, DA->get(F));

  // Rewrite function args and insert alloc_stack/dealloc_stack.
  //
  // WARNING: This may split critical edges in ValueLifetimeAnalysis.
  OpaqueStorageAllocation allocator(pass);
  allocator.allocateOpaqueStorage();

  DEBUG(llvm::dbgs() << "Finished allocating storage.\n"; F->dump();
        pass.valueStorageMap.dump());

  // Rewrite instructions with address-only operands or results.
  rewriteFunction(pass);

  deleteRewrittenInstructions(pass);

  StackNesting().correctStackNesting(F);

  // The CFG may change because of criticalEdge splitting during
  // createStackAllocation or StackNesting.
  invalidateAnalysis(F, SILAnalysis::InvalidationKind::BranchesAndInstructions);
}

/// The entry point to this function transformation.
void AddressLowering::run() {
  if (getModule()->getASTContext().LangOpts.EnableSILOpaqueValues) {
    for (auto &F : *getModule())
      runOnFunction(&F);
  }
  // Set the SIL state before the PassManager has a chance to run
  // verification.
  getModule()->setStage(SILStage::Lowered);
}

SILTransform *swift::createAddressLowering() { return new AddressLowering(); }
