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

//===----------------------------------------------------------------------===//
// Ownership Abstraction.
//
// FIXME: None of this should be in this pass. Ownership properties need an API
// apart from OwnershipCompatibilityUseChecker.
//===----------------------------------------------------------------------===//

// TODO: Figure out how to handle these, if possible.
static bool isUnknownUse(Operand *use) {
  switch (use->getUser()->getKind()) {
  default:
    return false;
  // mark_dependence requires recursion to find all uses. It should be
  // replaced by begin/end dependence..
  case SILInstructionKind::MarkDependenceInst:
  // select_enum propagates a value. We need a general API for instructions like
  // this.
  case SILInstructionKind::SelectEnumInst:
  // OwnershipVerifier says that ref_tail_addr, ref_to_raw_pointer, etc. can
  // accept an owned value, but don't consume it and appear to propagate
  // it. This shouldn't normally happen without a borrow.
  case SILInstructionKind::RefTailAddrInst:
  case SILInstructionKind::RefToRawPointerInst:
  case SILInstructionKind::RefToUnmanagedInst:
  case SILInstructionKind::RefToUnownedInst:
  // dynamic_method_br seems to capture self, presumably propagating lifetime.
  case SILInstructionKind::DynamicMethodBranchInst:
  // If a value is unsafely cast, we can't say anything about its lifetime.
  case SILInstructionKind::UncheckedBitwiseCastInst: // Is this right?
  case SILInstructionKind::UncheckedTrivialBitCastInst:
  // Ownership verifier says project_box can take an owned value, but
  // that doesn't make sense to me.
  case SILInstructionKind::ProjectBoxInst:
  case SILInstructionKind::ProjectExistentialBoxInst:
  // Ownership verifier says open_existential_box can take an owned value, but
  // that doesn't make sense to me.
  case SILInstructionKind::OpenExistentialBoxInst:
  // Unmanaged operations.
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
    llvm_unreachable("Unexpected use of a loadable owned value.");

  // Consume the value.
  case SILInstructionKind::AutoreleaseValueInst:
  case SILInstructionKind::DeallocBoxInst:
  case SILInstructionKind::DeallocExistentialBoxInst:
  case SILInstructionKind::DeallocRefInst:
  case SILInstructionKind::DeinitExistentialValueInst:
  case SILInstructionKind::DestroyValueInst:
  case SILInstructionKind::KeyPathInst:
  case SILInstructionKind::ReleaseValueInst:
  case SILInstructionKind::ReleaseValueAddrInst:
  case SILInstructionKind::StrongReleaseInst:
  case SILInstructionKind::StrongUnpinInst:
  case SILInstructionKind::UnownedReleaseInst:
  case SILInstructionKind::InitExistentialRefInst:
  case SILInstructionKind::InitExistentialValueInst:
  case SILInstructionKind::EndLifetimeInst:
  case SILInstructionKind::UnconditionalCheckedCastValueInst:
    return true;

  // Terminators must consume their values.
  case SILInstructionKind::BranchInst:
  case SILInstructionKind::CheckedCastBranchInst:
  case SILInstructionKind::CheckedCastValueBranchInst:
  case SILInstructionKind::CondBranchInst:
  case SILInstructionKind::ReturnInst:
  case SILInstructionKind::ThrowInst:
    return true;

  case SILInstructionKind::StoreInst:
    assert(cast<StoreInst>(user)->getSrc() == use->get());
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
  // EndBorrow extends the lifetime.
  case SILInstructionKind::EndBorrowInst:
    return false;

  // Extend the lifetime without borrowing, propagating, or destroying it.
  case SILInstructionKind::ClassMethodInst:
  case SILInstructionKind::DebugValueInst:
  case SILInstructionKind::ExistentialMetatypeInst:
  case SILInstructionKind::ValueMetatypeInst:
  case SILInstructionKind::BridgeObjectToWordInst:
  case SILInstructionKind::CopyBlockInst:
  case SILInstructionKind::FixLifetimeInst:
  case SILInstructionKind::SetDeallocatingInst:
  case SILInstructionKind::StoreWeakInst:
  case SILInstructionKind::StrongPinInst:
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

/// This pass' shared state per copied def.
struct CopyPropagationState {
  SILFunction *F;

  // Per-function invalidation state.
  unsigned invalidation;

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

  void markInvalid(SILAnalysis::InvalidationKind kind) {
    invalidation |= (unsigned)kind;
  }

  void reset(SILValue def) {
    clear();
    currDef = def;
  }

  // Do not clear invalidation. It accumulates for an entire function before
  // the PassManager is notified.
  void clear() {
    currDef = SILValue();
    liveBlocks.clear();
    lastUsers.clear();
    destroyBlocks.clear();
    blockDestroys.clear();
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

  Optional<bool> isLastUserConsuming(SILInstruction *user) {
    auto lastUseIter = lastUsers.find(user);
    if (lastUseIter == lastUsers.end())
      return None;
    return lastUseIter->second;
  }
  
  void setLastUse(Operand *use) {
    lastUsers[use->getUser()] |= isConsuming(use);
  }

  bool isDestroy(SILInstruction * inst) {
    auto destroyPos = blockDestroys.find(inst->getParent());
    return destroyPos != blockDestroys.end() && destroyPos->second != inst;
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

  SmallVector<SILBasicBlock *, 8> worklist({userBB});
  while (!worklist.empty()) {
    SILBasicBlock *bb = worklist.pop_back_val();
    for (auto *predBB : bb->getPredecessorBlocks()) {
      switch (pass.isBlockLive(predBB)) {
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
  auto isLive = pass.isBlockLive(bb);
  switch (isLive) {
  case LiveOut:
    return;
  case LiveWithin:
    pass.setLastUse(use);
    break;
  case Dead: {
    computeUseBlockLiveness(bb, pass);
    // If this block is in a loop, it will end up LiveOut.
    if (pass.isBlockLive(bb) == LiveWithin)
      pass.setLastUse(use);
    break;
  }
  }
}

/// Populate `pass.liveBlocks` and `pass.lastUsers`.
/// Return true if successful.
///
/// Assumptions: No users occur before 'def' in def's BB because this follows
/// the SSA def-use chains and all terminators consume their operand.
static bool computeLiveness(CopyPropagationState &pass) {
  SILBasicBlock *defBB;
  if (auto *defInst = pass.currDef->getDefiningInstruction())
    defBB = defInst->getParent();
  else
    defBB = cast<SILArgument>(pass.currDef)->getParent();

  pass.markBlockLive(defBB, LiveWithin);

  SmallSetVector<SILValue, 8> worklist;
  worklist.insert(pass.currDef);
  while (!worklist.empty()) {
    SILValue value = worklist.pop_back_val();
    for (Operand *use : value->getUses()) {
      auto *user = use->getUser();

      if (isUnknownUse(use)) {
        DEBUG(llvm::dbgs() << "Unknown owned value user: " << *user);
        return false;
      }
      if (auto *copy = dyn_cast<CopyValueInst>(user))
        worklist.insert(copy);

      // Skip begin_borrow. Consider its end_borrows the use points.
      if (auto *BBI = dyn_cast<BeginBorrowInst>(user)) {
        for (Operand *use : BBI->getUses()) {
          if (auto *EBI = dyn_cast<EndBorrowInst>(use->getUser()))
            visitUse(use, pass);
        }
      }
      if (isa<DestroyValueInst>(user)) {
        pass.destroyBlocks.insert(user->getParent());
        continue;
      }
      visitUse(use, pass);
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
  pass.blockDestroys[destroyBB] =
      B.createDestroyValue(succBB->begin()->getLoc(), pass.currDef);
  ++NumDestroysGenerated;
  pass.markInvalid(SILAnalysis::InvalidationKind::Instructions);
}

static void findOrInsertDestroyInBlock(SILBasicBlock *bb,
                                       CopyPropagationState &pass) {
  auto I = bb->getTerminator()->getIterator();
  while (true) {
    auto *inst = &*I;
    auto lastUserPos = pass.lastUsers.find(inst);
    if (lastUserPos == pass.lastUsers.end()) {
      // This is not a potential last user. Keep scanning.
      assert(I != bb->begin());
      --I;
      continue;
    }
    bool isConsume = lastUserPos->second;
    if (isConsume)
      pass.blockDestroys[bb] = inst;
    else {
      assert(inst != bb->getTerminator() && "Terminator must consume operand.");
      SILBuilderWithScope B(&*std::next(I));
      pass.blockDestroys[bb] =
          B.createDestroyValue(inst->getLoc(), pass.currDef);
      ++NumDestroysGenerated;
      pass.markInvalid(SILAnalysis::InvalidationKind::Instructions);
    }
    break;
  }
}

/// Populate `pass.blockDestroys` with the final destroy points once copies are
/// eliminated.
static void findOrInsertDestroys(CopyPropagationState &pass) {
  for (auto *destroyBB : pass.destroyBlocks) {
    // Backward CFG traversal.
    SmallSetVector<SILBasicBlock *, 8> worklist;
    auto visitBB = [&](SILBasicBlock *bb, SILBasicBlock *succBB) {
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
  assert(!isa<TermInst>(user) && "Terminator must consume its operand.");

  SILBuilder B(std::next(user->getIterator()));
  B.setCurrentDebugScope(user->getDebugScope());

  auto *copy = B.createCopyValue(user->getLoc(), use->get());
  use->set(copy);
  ++NumCopiesGenerated;
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
      if (!pass.isDestroy(destroy)) {
        instsToDelete.insert(destroy);
        ++NumDestroysEliminated;
      }
      return;
    }
    if (!pass.isDestroy(user))
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
      ++NumCopiesEliminated;
      continue;
    }
    for (Operand *use : value->getUses())
      visitUse(use);
  }
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
      if (auto *copy = dyn_cast<CopyValueInst>(&I))
        copiedDefs.insert(stripCopies(copy));
    }
  }
  for (auto &def : copiedDefs) {
    pass.reset(def);
    if (computeLiveness(pass)) {
      findOrInsertDestroys(pass);
      rewriteCopies(pass);
    }
  }
  invalidateAnalysis(SILAnalysis::InvalidationKind(pass.invalidation));
}

SILTransform *swift::createCopyPropagation() { return new CopyPropagation(); }
