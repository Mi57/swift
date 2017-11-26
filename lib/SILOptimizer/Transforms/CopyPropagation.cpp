//===--- OwnershipModelEliminator.cpp - Eliminate SILOwnership Instr. -----===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2017 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//
///
/// SSA Copy propagation removes unnecessary copy_value/destroy_value
/// instructions.
///
/// This requires ownership SSA form.
///
/// [WIP] This is an incomplete proof of concept. Ownership properties should
/// be exposed via an API that makes this pass trivially sound.
///
/// This is meant to complement -enable-sil-opaque-values. Initially this will
/// run at -O, but eventually may also be adapted to -Onone (but as currently
/// designed, it shrinks variable live ranges).
///
/// State:
/// copiedDefs : {SILValue}
/// liveBlocks : {SILBasicBlock -> [LiveOut|LiveWithin]}
/// lastUsers  : {SILInstruction -> bool isConsume} // potential last users
/// destroyBlocks: {SILBasicBlock} // original destroy points
/// blockDestroys: {SILBasicBlock -> SILInstruction} // final last users
///
/// 1. Forward walk the instruction stream. Record the originating source of any
///    copy_value into copiedDefs.
///
/// 2. For each copied Def, visit all uses:
///    - Recurse through copies.
///    - Skip over borrows.
///    - Ignore destroys as uses, add their block to destroyBlocks.
///
///    For each use, check it's block's liveness:
///    - If in liveBlocks and LiveOut: continue to the next use.
///    - If in liveBlocks and LiveWithin:
///        lastUsers.insert(Use), continue to next use.
///    - If not in liveBlocks, mark this block as LiveWithin
///      Then traverse CFG preds using a worklist, marking each live-out:
///      - If pred block is already in liveBlocks, set LiveOut and stop.
///
/// Observations:
/// - The current def must be postdominated by some subset of its destroys.
/// - The postdominating destroys cannot be within nested loops.
/// - Any blocks in nested loops are now marked LiveOut.
///
/// 3. Backward walk from destroyBlocks:
///    For each destroyBlock:
///    - if LiveOut: continue to next destroyBlock.
///    Follow predecessor blocks:
///    - if predBB is Dead: continue backward CFG traversal.
///    - if predBB is LiveOut: insert destroy in this CFG edge.
///    - if predBB is LiveWithin:
///        if not predBB in blockDestroys:
///          Backward scan predBB until I is in lastUsers.
///          If lastUser[I].isConsume:
///            blockDestroys[predBB] = I
///          else
///            blockDestroys[predBB] = new destroy_value.
///
/// 4. Revisit uses:
///    Remove copy_values in the def-use chain.
///    If a consumer is not in blockDestroys:
///      If it is a destroy, remove it.
///      Else copy the consumed element.
///
/// This is sound for ownership-SSA. Otherwise, we would need to handle
/// the same conditions as ValueLifetimeAnalysis.
///
/// TODO: This will only be an effective optimization for aggregates once SILGen
/// is no longer generating spurious borrows.
/// ===----------------------------------------------------------------------===

#define DEBUG_TYPE "copy-propagation"
#include "swift/SIL/Projection.h"
#include "swift/SILOptimizer/PassManager/Passes.h"
#include "swift/SILOptimizer/Utils/IndexTrie.h"

using namespace swift;
using llvm::SmallSetVector;
using llvm::PointerIntPair;

STATISIC(NumCopiesEliminated, "number of copy_value instructions removed");
STATISIC(NumDestroysEliminated, "number of destroy_value instructions removed");
STATISIC(NumCopiesGenerated, "number of copy_value instructions created");
STATISIC(NumDestroysGenerated, "number of destroy_value instructions created");

// FIXME: factor with AddressLowering if it ends up remaining useful.
namespace {
struct GetUser {
  SILInstruction *operator()(Operand *oper) const { return oper->getUser(); }
};
} // namespace

// TODO: LLVM needs a map_range.
static iterator_range<llvm::mapped_iterator<ValueBase::use_iterator, GetUser>>
getUserRange(SILValue val) {
  return make_range(llvm::map_iterator(val->use_begin(), GetUser()),
                    llvm::map_iterator(val->use_end(), GetUser()));
}

//===----------------------------------------------------------------------===//
// Ownership Abstraction.
//
// FIXME: None of this should be in this pass. Ownership properties need an API
// apart from OwnershipCompatibilityUseChecker.
//===----------------------------------------------------------------------===//

// TODO: Figure out how to handle these, if possible.
static bool isUnknownUse(Operand *use) {
  switch (use->getUser->getKind()) {
  default:
    return false;
  // mark_dependence requires recursion to find all uses. It should be
  // replaced by begin/end dependence..
  case MarkDependenceInst:
  // select_enum propagates a value. We need a general API for instructions like
  // this.
  case SelectEnumInst:
  // OwnershipVerifier says that ref_tail_addr, ref_to_raw_pointer, etc. can
  // accept an owned value, but don't consume it and appear to propagate it. This
  // shouldn't normally happen without a borrow.
  case RefTailAddrInst:
  case RefToRawPointerInst:
  case RefToUnmanagedInst:
  case RefToUnownedInst:
  // dynamic_method_br seems to capture self, presumably propagating lifetime.
  case DynamicMethodBranchInst:
  // If a value is unsafely cast, we can't say anything about its lifetime.
  case UncheckedBitwiseCastInst: // Is this right?
  case UncheckedTrivialBitCastInst:
  // Ownership verifier says project_box can take an owned value, but
  // that doesn't make sense to me.
  case ProjectBoxInst:
  case ProjectExistentialBoxInst:
  // Ownership verifier says open_existential_box can take an owned value, but
  // that doesn't make sense to me.
  case OpenExistentialBoxInst:
  // Unmanaged operations.
  case UnmanagedRetainValueInst:
  case UnmanagedReleaseValueInst:
  case UnmanagedAutoreleaseValueInst:
    return true;
  }
}

/// Return true if the given owned operand is consumed by the given call.
static bool isAppliedArgConsumed(ApplySite apply, Operand *oper) {
  ParameterConvention paramConv;
  if (oper->get() == apply->getCallee()) {
    assert(oper->getOperandNumber() == ApplySite::Callee &&
           "function can't be passed to itself");
    paramConv = apply.getSubstCalleeType()->getCalleeConvention();
  } else {
    unsigned argIndex = apply.getCalleeArgIndex(*oper);
    paramConv = apply.getSubstCalleeConv()
      .getParamInfoForSILArg(argIndex)
      .getConvention();
  }
  return isConsumedParameter(paramConv);
}

/// Return true if the given builtin consumes its operand.
static bool isBuiltinArgConsumed(BuiltinInst *BI) {
  const BuiltinInfo &Builtin = BI->getBuiltinInfo();
  switch (Builtin.ID) {
  default:
    llvm_unreachable("Unexpected Builtin with owned value operand.");
  // Extend lifetime without consuming.
  case ErrorInMain:
  case UnexpectedError:
  case WillThrow:
    return false;
  // UnsafeGuaranteed moves the value, which will later be destroyed.
  case UnsafeGuaranteed:
    return true;
  }
}

/// Return true if the given operand is consumed by its user.
/// 
/// TODO: Review the semantics of operations that extend the lifetime *without*
/// propagating the value. Ideally, that never happens without borrowing first.
static bool isConsuming(Operand *use) {
  auto *user = use->getUser();
  if (isa<ApplySite>(user))
    return isAppliedArgConsumed(ApplySite(user), use);

  if (auto *BI = dyn_cast<BuiltinInst>(user))
    return isBuiltinArgConsumed(BI);

  switch (user->getKind()) {
  default:
    llvm_unreachable("Unexpected last use of a loadable owned value.");

  // Consume the value.
  case AutoreleaseValueInst:
  case CheckedCastValueBranchInst:
  case DeallocBoxInst:
  case DeallocExistentialBoxInst:
  case DeallocRefInst:
  case DeinitExistentialValueInst:
  case DestroyValueInst:
  case KeyPathInst:
  case ReleaseValueInst:
  case ReleaseValueAddrInst:
  case StrongReleaseInst:
  case StrongUnpinInst:
  case UnownedReleaseInst:
  case InitExistentialRefInst:
  case InitExistentialValueInst:
  case EndLifetimeInst:
  case UnconditionalCheckedCastValueInst:
    return true;

  // Terminators must consume their values.
  case BranchInst:
  case CheckedCastBranchInst:
  case CheckedCastValueBranchInst:
  case CondBranchInst:
  case ReturnInst:
  case ThrowInst:
    return true;

  case StoreInst:
    assert(cast<StoreInst>(user)->getSrc() == use->get());
    return true;

  case DeallocPartialRefInst:
    return cast<DeallocPartialRefInst>(user)->getInstance() == use->get();

  // Move the value.
  case TupleInst:
  case StructInst:
  case ObjectInst:
  case EnumInst:
  case OpenExistentialRefInst:
  case UpcastInst:
  case UncheckedRefCastInst:
  case ConvertFunctionInst:
  case RefToBridgeObjectInst:
  case BridgeObjectToRefInst:
  case UnconditionalCheckedCastInst:
  case MarkUninitializedInst:
  case UncheckedEnumDataInst:
  case DestructureStructInst:
  case DestructureTupleInst:
    return true;

  // BeginBorrow should already be skipped.
  // EndBorrow extends the lifetime.  
  case EndBorrowInst:
    return false;

  // Extend the lifetime without borrowing, propagating, or destroying it.
  case ClassMethodInst:
  case DebugValueInst:
  case ExistentialMetatypeInst:
  case ValueMetatypeInst:
  case BridgeObjectToWordInst:
  case CopyBlockInst:
  case FixLifetimeInst:
  case SetDeallocatingInst:
  case StoreWeakInst:
  case StrongPinInst:
    return false;

  // Dynamic dispatch without capturing self.
  case ObjCMethodInst:
  case ObjCSuperMethodInst:
  case SuperMethodInst:
  case WitnessMethodInst:
    return false;
    
  }
}

//===----------------------------------------------------------------------===//
// CopyPropagationState: shared state for the pass's analysis and transforms.
//===----------------------------------------------------------------------===//

namespace {
enum IsLive_t { LiveWithin, LiveOut, Dead };

/// This pass' shared state per copied def.
CopyPropagationState {
  SILFunction *F;

  // Per-function invalidation state.
  Invalidation_t invalidation;

  // Current copied def for which this state describes the liveness.
  SILValue currDef;
  // Map of all blocks in which current def is live. True if it is also liveout.
  DenseMap<SILBasicBlock *, bool> liveBlocks;
  // Set of all last users in this def's live range and whether their used value
  // is consumed.
  DenseMap<SILInstruction *, bool> lastUsers;
  // Original points in the CFG where the current value was destroyed.
  DenseSet<SILBasicBlock *> destroyBlocks;
  // Map blocks that contain a final destroy to the destroying instruction.
  DenseMap<SILBasicBlock *, SILInstruction *> blockDestroys;

  CopyPropagationState(SILFunction * F)
      : F(F), invalidation(SILAnalysis::InvalidationKind::Nothing) {}

  void reset(SILValue def) {
    clear();
    currDef = def;
  }

  void clear() {
    currDef = SILValue();
    liveBlocks.clear();
    lastUsers.clear();
    destroyBlocks.clear();
    blockDestroys.clear();
  }

  BlockLive_t isBlockLive(SILBasicBlock *bb) const {
    auto &liveBlockIter = pass.liveBlocks.find(bb);
    if (liveBlockIter == pass.liveBlocks.end())
      return Dead;
    return liveBlockIter->second ? LiveOut : LiveWithin;
  }

  void markBlockLive(SILBasicBlock *bb, IsLive_t isLive) {
    assert(isLive != Dead && "erasing live blocks isn't implemented.");
    liveBlocks[bb] = isLive;
  }

  Optional<bool> isLastUserConsuming(SILInstruction *user) {
    auto &lastUseIter = lastUsers.find(user);
    if (lastUseIter == lastUsers.end())
      return None;
    return lastUseIter->second;
  }
  
  void setLastUse(Operand *use) {
    lastUsers[use->getUser()] |= isConsuming(use);
  }

  bool isDestroy(SILInstruction * inst) {
    auto destroyPos = blockDestroys.find(inst->getParent());
    return destroyPos != pass.blockDestroys.end() && destroyPos->second != inst;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Step 2. Find liveness for a copied def, ignoring copies and destroys.
// Populates pass.liveBlocks with LiveOut and LiveWithin blocks.
// Populates pass.lastUsers with potential last users.
//
// TODO: Make sure all dependencies are accounted for (mark_dependence,
// ref_element_addr, project_box?). We should have an ownership API so that the
// pass doesn't require any special knowledge of value dependencies.
//===----------------------------------------------------------------------===//

/// Mark blocks live in a reverse CFG traversal from this user.
static void computeUseBlockLiveness(SILBasicBlock *userBB,
                                    CopyPropagationState &pass) {

  pass.markBlockLive(userBB, LiveWithin);

  SmallVector<SILBasicBlock *, 8> worklist(userBB);
  while (!worklist.empty()) {
    SILBasicBlock *bb = worklist.pop_back_val();
    for (auto *predBB : bb->getPredecessorBlocks()) {
      switch (pass.isLive(predBB)) {
      case Dead:
        worklist.push_back(bb);
        LLVM_FALLTHROUGH;
      case LiveWithin:
        pass.markBlockLive(predBB, LiveOut);
        break;
      case LiveOut:
        break;
      }
    }
  }
}

/// Update the current def's liveness at the given user.
static void visitUse(Operand *use, CopyPropagationState &pass) {
  auto *bb = use->getUser()->getParent();
  auto isLive = pass.liveBlocks.isLive(bb);
  switch (isLive) {
  case LiveOut:
    return;
  case LiveWithin:
    pass.setLastUse(use);
  case Dead: {
    computeUseBlockLiveness(bb, pass);
    // If this block is in a loop, it will end up LiveOut.
    if (pass.liveBlocks.isLive(bb) == LiveWithin)
      pass.lastUsers.insert(user);
  }
  }
}

/// Populate `pass.liveBlocks` and `pass.lastUsers`.
/// Return true if successful.
///
/// Assumptions: No users occur before 'def' in def's BB because this follows
/// the SSA def-use chains and all terminators consume their operand.
static bool computeLiveness(SILValue def, CopyPropagationState &pass) {
  pass.markBlockLive(def.getBB(), LiveWithin);

  SmallSetVector<SILValue, 8> worklist(def);
  while (!worklist.empty()) {
    SILValue value = worklist.pop_back_val();
    for (Operand *use : value->getUses()) {
      auto *user = use->getUser();

      if (isUnknownUse(use)) {
        DEBUG(llvm::dbgs() << "Unknown owned value user: " << *user);
        return false;
      }
      if (isa<CopyValueInst>(user))
        worklist.insert(user);

      // Skip begin_borrow. Consider its end_borrows the use points.
      if (auto *BBI = dyn_cast<BeginBorrowInst>(user)) {
        for (Operand *use : BBI) {
          if (auto *EBI = dyn_cast<EndBorrowInst>(use->getUser()))
            pass.worklist.insert(EBI);
        }
      }
      if (isa<DestroyValueInst>(user)) {
        pass.destroyBlocks.insert(user->getParentBB());
        continue;
      }
      visitUse(use);
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Step 3. Find the destroy points of the current value.
//===----------------------------------------------------------------------===//

static void insertDestroyOnCFGEdge(SILBasicBlock *predBB, SILBasicBlock *succBB,
                                   CopyPropagationState &pass) {
  auto *destroyBB = splitIfCriticalEdge(predBB, succBB);
  if (destroyBB != succBB)
    pass.invalidation |= SILAnalysis::InvalidationKind::Branches;

  SILBuilderWithScope B(destroyBB->begin());
  pass.blockDestroys[destroyBB] =
      B.createDestroyValue(succBB->begin()->getLoc(), pass.currDef);
  pass.invalidation |= SILAnalysis::InvalidationKind::Instructions;
}

static void findOrInsertDestroyInBlock(SILBasicBlock *bb,
                                       CopyPropagationState &pass) {
  auto I = bb->getTerminator()->getIterator;
  while (true) {
    auto *inst = &*I;
    auto lastUserPos = pass.lastUsers.find(inst);
    if (lastUserPos == pass.lastUsers.end()) {
      // This is not a potential last user. Keep scanning.
      assert(I != bb->begin());
      --I;
      continue;
    }
    bool isConsume = lastUserIt->second;
    if (isConsume)
      blockDestroys[bb] = inst;
    else {
      assert(inst != bb->getTerminator() && "Terminator must consume operand.");
      SILBuilderWithScope B(&*next(I));
      blockDestroys[bb] = B.createDestroyValue(inst->getLoc(), currDef);
      pass.invalidation |= SILAnalysis::InvalidationKind::Instructions;
    }
    break;
  }
}

/// Populate `pass.blockDestroys` with the final destroy points once copies are
/// eliminated.
static void findOrInsertDestroys(CopyPropagationState &pass) {
  for (auto *destroyBB : pass.destroyBlocks) {
    // Backward CFG traversal.
    SmallSetVector<SILBasicBlock *> worklist;
    auto visitBB = [&](SILBasicBlock *bb, *succBB) {
      switch (pass.isBlockLive(bb)) {
      case LiveOut:
        // If succBB is null, then the destroy in destroyBB must be an inner
        // nested destroy. Skip it.
        if (succBB)
          insertDestroyOnCFGEdge(bb, succBB, pass);
        break;
      case LiveWithin:
        findOrInsertDestroyInBlock(bb, pass);
        break;
      case Dead: {
        worklist.insert(bb);
      }
      }
    };
    visitBB(destroyBB, nullptr);
    while (!worklist.empty()) {
      auto *succBB = worklist.pop_back_val();
      for (auto *predBB : succBB->getPredecessorBlocks())
        visitBB(predBB, succBB);
    }
  }
}

//===----------------------------------------------------------------------===//
// Step 4. Rewrite copies and destroys for a single copied definition.
//===----------------------------------------------------------------------===//

/// The current value must be live across the given use. Copy the value for this
/// use if necessary.
static void copyLiveUse(Operand *use, CopyPropagationState &pass) {
  if (!isConsuming(use))
    return;

  SILInstruction *user = use->getUser();
  assert(!isa<TerminatorInst>(user) && "Terminator must consume its operand.");

  SILBuilder B(std::next(user->getIterator()));
  B.setDebugScope(user);

  auto *copy = B.createCopyValue(user->getLoc(), use->get());
  use->set(copy);
  ++NumCopiesGenerated;
}

// TODO: Avoid churn. Identify destroys that already complement a last use.
static void rewriteCopies(SILValue def, CopyPropagationState &pass) {
  SmallSetVector<SILInstruction *, 8> instsToDelete;
  SmallSetVector<SILValue, 8> worklist(def);

  while (!worklist.empty()) {
    SILValue value = worklist.pop_back_val();
    if (auto *copy = dyn_cast<CopyValueInst>(value)) {
      worklist.append(getUserRange(value));
      copy->replaceAllUsesWith(copy->getOperand());
      instsToDelete.insert(copy);
      ++NumCopiesEliminated;
      continue;
    }
    for (Operand *use : value->getUses()) {
      auto *user = use->getUser();
      if (isa<CopyValueInst>(user)) {
        worklist.insert(user);
        continue;
      }
      if (isa<DestroyValueInst>(user)) {
        if (!pass.isDestroy(user)) {
          instsToDelete.insert(user);
          ++NumDestroysEliminated;
        }
        continue;
      }
      if (!pass.isDestoy(user))
        copyLiveUse(use, pass);
    }
  }
#ifndef NDEBUG
  for (auto *lastUser : lastUsers)
    llvm::dbgs() << "Unreachable last user: " << *lastUser;
#endif
  assert(pass.lastUsers.empty() && "Unreachable last user");

  if (!instsToDelete.empty()) {
    recursivelyDeleteTriviallyDeadInstructions(pass.instsToDelete.takeVector(),
                                               /*force=*/true);
    pass.invalidation |= SILAnalysis::InvalidationKind::Instructions;
  }
}

//===----------------------------------------------------------------------===//
// CopyPropagation: Top-Level Function Transform.
//===----------------------------------------------------------------------===//

/// TODO: we could strip casts as well, then when recursing through users keep
/// track of the nearest non-copy def. For opaque values, we don't expect to see
/// casts.
static SILValue stripCopies(SILValue v) {
  while (true) {
    v = stripSinglePredecessorArgs(v);

    if (auto *srcCopy = dyn_cast<CopyValueInst>(v)) {
      v = srcCopy->getOperand();
      continue;
    }
    return v;
  }
}

namespace {
class CopyPropagation : public SILModuleTransform {
  /// A trie of integer indices that allows identificaiton of a projection
  /// path. There only needs to be one of these for all passes in a module, but
  /// each pass currently defines its own.
  IndexTrieNode *subPathTrie;

  CopyPropagation(): subPathTrie(nullptr) {}

  ~CopyPropagation() {
    delete subPathTrie;
  }

  /// The entry point to this function transformation.
  void run() override;
};
} // end anonymous namespace

/// Top-level pass driver.
///
/// Step 1. Find all copied defs.
/// Then invoke the entry points for
/// Step 2: computeLiveness
/// Step 3: findDestroys
/// Step 4: rewriteCopies
void CopyPropagation::run() {
  DEBUG(llvm::dbgs() << "*** CopyPropagation: " << F.getName() << "\n");

  assert(getFunction()->hasQualifiedOwnership()
         && "SSA copy propagation is only valid for ownership-SSA");
  
  CopyPropagationState pass(getFunction());
  DenseMap<SILValue> copiedDefs;
  for (auto &BB : *pass.F) {
    for (auto &I : BB) {
      if (auto *copy = dyn_cast<CopyValueInst>(&I))
        copiedDefs.insert(stripCopies(copy));
    }
  }
  for (auto &def : copiedDefs) {
    pass.reset(def);
    if (computeLiveness(def, pass)) {
      findOrInsertDestroys(pass);
      rewriteCopies(def, pass);
    }
  }
  invalidateAnalysis(pass.invalidation);
}

SILTransform *swift::createCopyPropagation() { return new CopyPropagation(); }
