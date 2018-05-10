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
/// users      : {SILInstruction -> bool isConsume} // potential last users
/// destroyBlocks: {SILBasicBlock} // original destroy points
/// blockDestroys: {SILBasicBlock -> SILInstruction} // final last users
///
/// 1. Forward walk the instruction stream. Record the originating source of any
///    copy_value into copiedDefs.
///
/// 2. For each copied Def, visit all uses:
///    - Recurse through copies.
///    - Skip over borrows.
///    - Ignore destroys as uses, but add their block to destroyBlocks.
///
///    For each use:
///    - If the Use is consumed: mark add its block to destroyBlocks.
///    Check the use block's liveness:
///    - If in liveBlocks and LiveOut: continue to the next use.
///    - If in liveBlocks and LiveWithin:
///        users.insert(Use), continue to next use.
///    - If not in liveBlocks, mark this block as LiveWithin
///      Then traverse CFG preds using a worklist, marking each live-out:
///      - If pred block is already in liveBlocks, set LiveOut and stop.
///      If the block is still in LiveWithin: users.insert(Use).
///
/// Observations:
/// - The current def must be postdominated by some subset of its
///   consuming uses, including destroys.
/// - The postdominating consumes cannot be within nested loops.
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
///          Backward scan predBB until I is in users.
///          If users[I].isConsume:
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
///
/// TODO: If we can't remove copies of borrows, then adapt this to handle +0
/// values. Essentially: don't mark anything a final destroy and avoid inserting
/// the new destroys.
///
/// TODO: Delete instructions with no side effects that produce values which are
/// immediately destroyed after copy propagation.
/// ===----------------------------------------------------------------------===

#define DEBUG_TYPE "copy-propagation"
#include "swift/SIL/InstructionUtils.h"
#include "swift/SIL/Projection.h"
#include "swift/SILOptimizer/PassManager/Passes.h"
#include "swift/SILOptimizer/PassManager/Transforms.h"
#include "swift/SILOptimizer/Utils/CFG.h"
#include "swift/SILOptimizer/Utils/IndexTrie.h"
#include "swift/SILOptimizer/Utils/Local.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Statistic.h"

using namespace swift;
using llvm::DenseMap;
using llvm::DenseSet;
using llvm::SmallSetVector;

STATISTIC(NumCopiesEliminated, "number of copy_value instructions removed");
STATISTIC(NumDestroysEliminated,
          "number of destroy_value instructions removed");
STATISTIC(NumCopiesGenerated, "number of copy_value instructions created");
STATISTIC(NumDestroysGenerated, "number of destroy_value instructions created");
STATISTIC(NumUnknownUsers, "number of functions with unknown users");

//===----------------------------------------------------------------------===//
// Ownership Abstraction.
//
// FIXME: None of this should be in this pass. Ownership properties need an API
// apart from OwnershipCompatibilityUseChecker.
//
// Users of owned values:
// U1. Instantaneous borrow, or "normal use" (copy_value, @guaranteed)
// U2. Escape an address to the value (ref_to_unowned,
// unchecked_trivial_bitcast) U3. Propagate the value without consuming it
// (mark_dependence, begin_borrow) U4. Consume the value immediately (store,
// destroy, @owned) U5. Consume the value indirectly via a move (tuple, struct)
//===----------------------------------------------------------------------===//

// TODO: Figure out how to handle these, if possible.
static bool isUnknownUse(Operand *use) {
  switch (use->getUser()->getKind()) {
  default:
    return false;
  // FIXME: (U3) mark_dependence requires recursion to find all uses. It should
  // be replaced by begin/end dependence.
  case SILInstructionKind::MarkDependenceInst: // Dependent
  // FIXME: (U3) ref_tail_addr should require a borrow because it doesn't have
  // fix_lifetime like other escaping instructions.
  case SILInstructionKind::RefTailAddrInst:
  // FIXME: (U3) dynamic_method_br seems to capture self, presumably propagating
  // lifetime. This should probably borrow self, then be treated like
  // mark_dependence.
  case SILInstructionKind::DynamicMethodBranchInst:
  // FIXME: (U3) Ownership verifier says project_box can accept an owned value
  // as a normal use, but it projects the address.
  case SILInstructionKind::ProjectBoxInst:
  case SILInstructionKind::ProjectExistentialBoxInst:
  // FIXME: (U3) Ownership verifier says open_existential_box can accept an
  // owned value as a normal use, but it projects an address.
  case SILInstructionKind::OpenExistentialBoxInst:
  // Unmanaged operations hopefully don't apply to the same value as CopyValue?
  case SILInstructionKind::UnmanagedRetainValueInst:
  case SILInstructionKind::UnmanagedReleaseValueInst:
  case SILInstructionKind::UnmanagedAutoreleaseValueInst:
    return true;
  }
}

/// Return true if the given owned operand is consumed by the given call.
static bool isAppliedArgConsumed(ApplySite apply, Operand *oper) {
  ParameterConvention paramConv;
  if (oper->get() == apply.getCallee()) {
    assert(oper->getOperandNumber() == 0
           && "function can't be passed to itself");
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
  case BuiltinValueKind::ErrorInMain:
  case BuiltinValueKind::UnexpectedError:
  case BuiltinValueKind::WillThrow:
    return false;
  // UnsafeGuaranteed moves the value, which will later be destroyed.
  case BuiltinValueKind::UnsafeGuaranteed:
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
    llvm::dbgs() << *user;
    llvm_unreachable("Unexpected use of a loadable owned value.");

  // Consume the value.
  case SILInstructionKind::AutoreleaseValueInst:
  case SILInstructionKind::DeallocBoxInst:
  case SILInstructionKind::DeallocExistentialBoxInst:
  case SILInstructionKind::DeallocRefInst:
  case SILInstructionKind::DeinitExistentialValueInst:
  case SILInstructionKind::DestroyValueInst:
  case SILInstructionKind::EndLifetimeInst:
  case SILInstructionKind::InitExistentialRefInst:
  case SILInstructionKind::InitExistentialValueInst:
  case SILInstructionKind::KeyPathInst:
  case SILInstructionKind::ReleaseValueInst:
  case SILInstructionKind::ReleaseValueAddrInst:
  case SILInstructionKind::StoreInst:
  case SILInstructionKind::StrongReleaseInst:
  case SILInstructionKind::StrongUnpinInst:
  case SILInstructionKind::UnownedReleaseInst:
  case SILInstructionKind::UnconditionalCheckedCastValueInst:
    return true;

  // Terminators must consume their owned values.
  case SILInstructionKind::BranchInst:
  case SILInstructionKind::CheckedCastBranchInst:
  case SILInstructionKind::CheckedCastValueBranchInst:
  case SILInstructionKind::CondBranchInst:
  case SILInstructionKind::ReturnInst:
  case SILInstructionKind::ThrowInst:
    return true;

  case SILInstructionKind::DeallocPartialRefInst:
    return cast<DeallocPartialRefInst>(user)->getInstance() == use->get();

  // Move the value.
  case SILInstructionKind::TupleInst:
  case SILInstructionKind::StructInst:
  case SILInstructionKind::ObjectInst:
  case SILInstructionKind::EnumInst:
  case SILInstructionKind::OpenExistentialRefInst:
  case SILInstructionKind::UpcastInst:
  case SILInstructionKind::UncheckedRefCastInst:
  case SILInstructionKind::ConvertFunctionInst:
  case SILInstructionKind::RefToBridgeObjectInst:
  case SILInstructionKind::BridgeObjectToRefInst:
  case SILInstructionKind::UnconditionalCheckedCastInst:
  case SILInstructionKind::MarkUninitializedInst:
  case SILInstructionKind::UncheckedEnumDataInst:
  case SILInstructionKind::DestructureStructInst:
  case SILInstructionKind::DestructureTupleInst:
    return true;

  // BeginBorrow should already be skipped.
  // EndBorrow extends the lifetime like a normal use.
  case SILInstructionKind::EndBorrowInst:
    return false;

  // Extend the lifetime without borrowing, propagating, or destroying it.
  case SILInstructionKind::BridgeObjectToWordInst:
  case SILInstructionKind::ClassMethodInst:
  case SILInstructionKind::CopyBlockInst:
  case SILInstructionKind::CopyValueInst:
  case SILInstructionKind::DebugValueInst:
  case SILInstructionKind::ExistentialMetatypeInst:
  case SILInstructionKind::FixLifetimeInst:
  case SILInstructionKind::SelectEnumInst:
  case SILInstructionKind::SetDeallocatingInst:
  case SILInstructionKind::StoreWeakInst:
  case SILInstructionKind::StrongPinInst:
  case SILInstructionKind::ValueMetatypeInst:
    return false;

  // Escape the value. The lifetime must already be enforced via something like
  // fix_lifetime.
  case SILInstructionKind::RefToRawPointerInst:
  case SILInstructionKind::RefToUnmanagedInst:
  case SILInstructionKind::RefToUnownedInst:
  case SILInstructionKind::UncheckedBitwiseCastInst:
  case SILInstructionKind::UncheckedTrivialBitCastInst:
    return false;

  // Dynamic dispatch without capturing self.
  case SILInstructionKind::ObjCMethodInst:
  case SILInstructionKind::ObjCSuperMethodInst:
  case SILInstructionKind::SuperMethodInst:
  case SILInstructionKind::WitnessMethodInst:
    return false;
    
  }
}

//===----------------------------------------------------------------------===//
// CopyPropagationState: shared state for the pass's analysis and transforms.
//===----------------------------------------------------------------------===//

namespace {
enum IsLiveKind { LiveWithin, LiveOut, Dead };

/// Liveness information produced by step #2, computeLiveness, and consumed by
/// step #3, findOrInsertDestroys.
class LivenessInfo {
  // Map of all blocks in which current def is live. True if it is also liveout.
  DenseMap<SILBasicBlock *, bool> liveBlocks;
  // Set of all "interesting" users in this def's live range and whether their
  // used value is consumed.
  DenseMap<SILInstruction *, bool> users;
  // Original points in the CFG where the current value was destroyed.
  typedef SmallSetVector<SILBasicBlock *, 8> BlockSetVec;
  BlockSetVec destroyBlocks;

public:
  bool empty() {
    assert(!liveBlocks.empty() || users.empty());
    return liveBlocks.empty();
  }

  void clear() {
    liveBlocks.clear();
    users.clear();
    destroyBlocks.clear();
  }

  IsLiveKind isBlockLive(SILBasicBlock *bb) const {
    auto liveBlockIter = liveBlocks.find(bb);
    if (liveBlockIter == liveBlocks.end())
      return Dead;
    return liveBlockIter->second ? LiveOut : LiveWithin;
  }

  void markBlockLive(SILBasicBlock *bb, IsLiveKind isLive) {
    assert(isLive != Dead && "erasing live blocks isn't implemented.");
    liveBlocks[bb] = isLive;
  }

  Optional<bool> isConsumingUser(SILInstruction *user) const {
    auto useIter = users.find(user);
    if (useIter == users.end())
      return None;
    return useIter->second;
  }

  void recordUser(Operand *use) { users[use->getUser()] |= isConsuming(use); }

  void recordOriginalDestroy(Operand *use) {
    destroyBlocks.insert(use->getUser()->getParent());
  }

  llvm::iterator_range<BlockSetVec::const_iterator> getDestroyBlocks() const {
    return destroyBlocks;
  }
};

/// Destroy information produced by step #3, findOrInsertDestroys, and consumed
/// by step #4, rewriteCopies.
///
/// This remains valid during copy rewriting. The only instructions referenced
/// are destroys that cannot be deleted.
class DestroyInfo {
  // Map blocks that contain a final destroy to the destroying instruction.
  DenseMap<SILBasicBlock *, SILInstruction *> blockDestroys;

public:
  bool empty() const { return blockDestroys.empty(); }

  void clear() { blockDestroys.clear(); }

  bool hasDestroy(SILBasicBlock *bb) const { return blockDestroys.count(bb); }

  // Return true if this instruction is marked as a final destroy point of the
  // current def's live range. A destroy can only be claimed once because
  // instructions like `tuple` can consume the same value via multiple operands.
  bool claimDestroy(SILInstruction *inst) {
    auto destroyPos = blockDestroys.find(inst->getParent());
    if (destroyPos != blockDestroys.end() && destroyPos->second == inst) {
      blockDestroys.erase(destroyPos);
      return true;
    }
    return false;
  }

  void recordFinalDestroy(SILInstruction *inst) {
    blockDestroys[inst->getParent()] = inst;
  }

  void invalidateFinalDestroy(SILInstruction *inst) {
    blockDestroys[inst->getParent()] = inst;
  }
};

/// This pass' shared state.
struct CopyPropagationState {
  SILFunction *F;

  // Per-function invalidation state.
  unsigned invalidation;

  // Current copied def for which this state describes the liveness.
  SILValue currDef;

  // computeLiveness result.
  LivenessInfo liveness;

  // findOrInsertDestroys result.
  DestroyInfo destroys;

  CopyPropagationState(SILFunction *F)
      : F(F), invalidation(SILAnalysis::InvalidationKind::Nothing) {}

  void markInvalid(SILAnalysis::InvalidationKind kind) {
    invalidation |= (unsigned)kind;
  }

  void resetDef(SILValue def) {
    // Do not clear invalidation. It accumulates for an entire function before
    // the PassManager is notified.
    liveness.clear();
    destroys.clear();
    currDef = def;
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Step 2. Find liveness for a copied def, ignoring copies and destroys.
//
// Generate pass.liveness.
// - mark blocks as LiveOut or LiveWithin.
// - record users that may become the last user on that path.
// - record blocks in which currDef may be destroyed.
//
// TODO: Make sure all dependencies are accounted for (mark_dependence,
// ref_element_addr, project_box?). We should have an ownership API so that the
// pass doesn't require any special knowledge of value dependencies.
//===----------------------------------------------------------------------===//

/// Mark blocks live in a reverse CFG traversal from this user.
static void computeUseBlockLiveness(SILBasicBlock *userBB,
                                    CopyPropagationState &pass) {

  pass.liveness.markBlockLive(userBB, LiveWithin);

  SmallVector<SILBasicBlock *, 8> worklist({userBB});
  while (!worklist.empty()) {
    SILBasicBlock *bb = worklist.pop_back_val();
    for (auto *predBB : bb->getPredecessorBlocks()) {
      switch (pass.liveness.isBlockLive(predBB)) {
      case Dead:
        worklist.push_back(bb);
        LLVM_FALLTHROUGH;
      case LiveWithin:
        pass.liveness.markBlockLive(predBB, LiveOut);
        break;
      case LiveOut:
        break;
      }
    }
  }
}

/// Update the current def's liveness at the given user.
///
/// Terminators consume their owned operands, so they are not live out of the
/// block.
static void computeUseLiveness(Operand *use, CopyPropagationState &pass) {
  auto *bb = use->getUser()->getParent();
  auto isLive = pass.liveness.isBlockLive(bb);
  switch (isLive) {
  case LiveOut:
    return;
  case LiveWithin:
    pass.liveness.recordUser(use);
    break;
  case Dead: {
    computeUseBlockLiveness(bb, pass);
    // If this block is in a loop, it will end up LiveOut.
    if (pass.liveness.isBlockLive(bb) == LiveWithin)
      pass.liveness.recordUser(use);
    break;
  }
  }
}

/// Generate pass.liveness.
/// Return true if successful.
///
/// Assumption: No users occur before 'def' in def's BB because this follows the
/// SSA def-use chains and all terminators consume their operand if it is owned.
static bool computeLiveness(CopyPropagationState &pass) {
  assert(pass.liveness.empty());

  SILBasicBlock *defBB;
  if (auto *defInst = pass.currDef->getDefiningInstruction())
    defBB = defInst->getParent();
  else
    defBB = cast<SILArgument>(pass.currDef)->getParent();

  pass.liveness.markBlockLive(defBB, LiveWithin);

  SmallSetVector<SILValue, 8> worklist;
  worklist.insert(pass.currDef);
  while (!worklist.empty()) {
    SILValue value = worklist.pop_back_val();
    for (Operand *use : value->getUses()) {
      auto *user = use->getUser();

      if (isUnknownUse(use)) {
        DEBUG(llvm::dbgs() << "Unknown owned value user: "; user->dump());
        ++NumUnknownUsers;
        return false;
      }
      if (auto *copy = dyn_cast<CopyValueInst>(user)) {
        worklist.insert(copy);
        continue;
      }
      // Skip begin_borrow. Consider its end_borrows the use points.
      if (auto *BBI = dyn_cast<BeginBorrowInst>(user)) {
        for (Operand *use : BBI->getUses()) {
          if (auto *EBI = dyn_cast<EndBorrowInst>(use->getUser()))
            computeUseLiveness(use, pass);
        }
        continue;
      }
      if (isConsuming(use)) {
        pass.liveness.recordOriginalDestroy(use);
        // Destroying a values does not force liveness.
        if (isa<DestroyValueInst>(user))
          continue;
      }
      computeUseLiveness(use, pass);
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
    pass.markInvalid(SILAnalysis::InvalidationKind::Branches);

  SILBuilderWithScope B(destroyBB->begin());
  auto *DI = B.createDestroyValue(succBB->begin()->getLoc(), pass.currDef);
  pass.destroys.recordFinalDestroy(DI);
  ++NumDestroysGenerated;
  DEBUG(llvm::dbgs() << "  Destroy on edge "; DI->dump());
  pass.markInvalid(SILAnalysis::InvalidationKind::Instructions);
}

static void insertDestroyAtInst(SILBasicBlock::iterator pos,
                                CopyPropagationState &pass) {
  SILBuilderWithScope B(pos);
  auto *DI = B.createDestroyValue((*pos).getLoc(), pass.currDef);
  pass.destroys.recordFinalDestroy(DI);
  ++NumDestroysGenerated;
  DEBUG(llvm::dbgs() << "  Destroy at last use "; DI->dump());
  pass.markInvalid(SILAnalysis::InvalidationKind::Instructions);
}

static void findOrInsertDestroyInBlock(SILBasicBlock *bb,
                                       CopyPropagationState &pass) {
  auto *defInst = pass.currDef->getDefiningInstruction();
  auto I = bb->getTerminator()->getIterator();
  while (true) {
    auto *inst = &*I;
    Optional<bool> isConsumingResult = pass.liveness.isConsumingUser(inst);
    if (isConsumingResult.hasValue()) {
      if (isConsumingResult.getValue()) {
        // This consuming use becomes a final destroy.
        pass.destroys.recordFinalDestroy(inst);
        break;
      }
      // Insert a destroy after this non-consuming use.
      assert(inst != bb->getTerminator() && "Terminator must consume operand.");
      insertDestroyAtInst(std::next(I), pass);
      break;
    }
    // This is not a potential last user. Keep scanning.
    // If the original destroy is reached, this is a dead live range. Insert a
    // destroy immediately after the def.
    if (I == bb->begin()) {
      assert(cast<SILArgument>(pass.currDef)->getParent() == bb);
      insertDestroyAtInst(I, pass);
      break;
    }
    --I;
    if (&*I == defInst) {
      insertDestroyAtInst(std::next(I), pass);
      break;
    }
  }
}

/// Populate `pass.blockDestroys` with the final destroy points once copies are
/// eliminated.
static void findOrInsertDestroys(CopyPropagationState &pass) {
  assert(!pass.liveness.empty());
  for (auto *destroyBB : pass.liveness.getDestroyBlocks()) {
    // Backward CFG traversal.
    SmallSetVector<SILBasicBlock *, 8> worklist;
    auto visitBB = [&](SILBasicBlock *bb, SILBasicBlock *succBB) {
      switch (pass.liveness.isBlockLive(bb)) {
      case LiveOut:
        // If succBB is null, then the destroy in destroyBB must be an inner
        // nested destroy. Skip it.
        if (succBB)
          insertDestroyOnCFGEdge(bb, succBB, pass);
        break;
      case LiveWithin:
        if (!pass.destroys.hasDestroy(bb))
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
  SILBuilder B(user->getIterator());
  B.setCurrentDebugScope(user->getDebugScope());

  auto *copy = B.createCopyValue(user->getLoc(), use->get());
  use->set(copy);
  ++NumCopiesGenerated;
  DEBUG(llvm::dbgs() << "  Copying at last use "; copy->dump());
  pass.markInvalid(SILAnalysis::InvalidationKind::Instructions);
}

// TODO: Avoid churn. Identify copies and destroys that already complement a
// non-consuming use.
static void rewriteCopies(CopyPropagationState &pass) {
  SmallSetVector<SILInstruction *, 8> instsToDelete;
  SmallSetVector<SILValue, 8> worklist;

  auto visitUse = [&](Operand *use) {
    auto *user = use->getUser();
    if (auto *copy = dyn_cast<CopyValueInst>(user)) {
      worklist.insert(copy);
      return;
    }
    if (auto *destroy = dyn_cast<DestroyValueInst>(user)) {
      if (!pass.destroys.claimDestroy(destroy)) {
        instsToDelete.insert(destroy);
        DEBUG(llvm::dbgs() << "  Removing "; destroy->dump());
        ++NumDestroysEliminated;
      }
      return;
    }
    if (!pass.destroys.claimDestroy(user))
      copyLiveUse(use, pass);
  };

  worklist.insert(pass.currDef);
  while (!worklist.empty()) {
    SILValue value = worklist.pop_back_val();
    // Recurse through copies then remove them.
    if (auto *copy = dyn_cast<CopyValueInst>(value)) {
      for (auto *use : copy->getUses())
        visitUse(use);
      copy->replaceAllUsesWith(copy->getOperand());
      instsToDelete.insert(copy);
      DEBUG(llvm::dbgs() << "  Removing "; copy->dump());
      ++NumCopiesEliminated;
      continue;
    }
    for (Operand *use : value->getUses())
      visitUse(use);
  }
  assert(pass.destroys.empty());

  if (!instsToDelete.empty()) {
    recursivelyDeleteTriviallyDeadInstructions(instsToDelete.takeVector(),
                                               /*force=*/true);
    pass.markInvalid(SILAnalysis::InvalidationKind::Instructions);
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
class CopyPropagation : public SILFunctionTransform {
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
  DEBUG(llvm::dbgs() << "*** CopyPropagation: " << getFunction()->getName()
                     << "\n");

  CopyPropagationState pass(getFunction());
  SmallSetVector<SILValue, 16> copiedDefs;
  for (auto &BB : *pass.F) {
    for (auto &I : BB) {
      if (auto *copy = dyn_cast<CopyValueInst>(&I)) {
        if (pass.F->hasQualifiedOwnership()
            || copy->getType().isOpaque(pass.F->getModule())) {
          copiedDefs.insert(stripCopies(copy));
        }
      }
    }
  }
  for (auto &def : copiedDefs) {
    if (def.getOwnershipKind() != ValueOwnershipKind::Owned)
      continue;

    pass.resetDef(def);
    if (computeLiveness(pass)) {
      findOrInsertDestroys(pass);
      // Invalidate book-keeping before deleting instructions.
      pass.liveness.clear();
      rewriteCopies(pass);
    }
  }
  invalidateAnalysis(SILAnalysis::InvalidationKind(pass.invalidation));
}

SILTransform *swift::createCopyPropagation() { return new CopyPropagation(); }
