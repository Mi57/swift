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
/// be exposed via an API that makes this pass trivially complete.
///
/// This is meant to complement opaque values. Initially this will run at -O,
/// but eventually may also be adapted to -Onone (as currently designed, it
/// shrinks variable live ranges).
///
/// State:
/// copiedDefs : {SILValue}
/// liveBlocks : {SILBasicBlock, bool isLiveOut}
/// lastUsers  : {Operand}
///
/// 1. Forward walk the instruction stream. Record the originating source of any
///    copy_value into copiedDefs.
///
/// 2. For each copied Def, visit all uses:
///    - Recurse through copies.
///    - Skip over borrows.
///    - Ignore destroys.
///
///    For each use:
///    - If in liveBlocks and isLiveOut, continue to the next use.
///    - If in liveBlocks and !liveout, scan backward from this Use:
///      - If lastUsers.erase(I); lastUsers.insert(Use), continue to next use.
///    - Mark this block as live, not isLiveOut.
///    Then traverse CFG preds using a worklist, marking each live-out:
///    - If pred block is already in liveBlocks, set isLiveOut and stop.
///
/// 3. Revisit uses:
///    If a consumer is not the last use, copy the consumed element.
///    If a last use is consuming, remove the destroy of the consumed element.
///
/// This is sound assuming for ownership-SSA. Otherwise, we would need to handle
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

/// This pass' shated state per copied def.
CopyPropagationState {
  SILFunction *F;

  // Map of all blocks in which current def is live. True if it is also liveout.
  DenseMap<SILBasicBlock *, bool> liveBlocks;
  // Set of all last users in this def's live range.
  DenseSet<SILInstruction *> lastUsers;

  CopyPropagationState(SILFunction *F): F(F) {

  void clear() {
    liveBlocks.clear();
    lastUsers.clear();
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
};
} // namespace

//===----------------------------------------------------------------------===//
// Find liveness and last users ignoring copies and destroys.
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

/// Scan this user's block for another lastUser. If found, replace it.
///
/// TODO: This could be costly for many users in a very large block. Consider
/// using a ValueLifetime-like approach of finding live blocks first before last
/// users. The current approach has the advantage of not storing UserSet.
static void findLastUser(SILInstruction *user, CopyPropagationState &pass) {
  auto *I = user->getIterator();
  auto *B = user->getParent()->begin;
  while (I != B) {
    --I;
    if (pass.lastUsers.erase(&*I)) {
      pass.lastUsers.insert(&*I);
      return;
    }
  }
}

/// Update the current def's liveness at the given user.
static void visitUser(SILInstruction *user, CopyPropagationState &pass) {
  auto *bb = user->getParent();
  auto isLive = pass.liveBlocks.isLive(bb);
  switch (isLive) {
  case LiveOut:
    return;
  case LiveWithin:
    findLastUser(user, pass);
  case Dead:
    computeUseBlockLiveness(bb, pass);
  }
}

/// Populate pass.liveBlocks and pass.lastUsers.
static bool findUsers(SILValue def, CopyPropagationState &pass) {
  SmallSetVector<SILValue, 8> worklist(def);

  while (!worklist.empty()) {
    SILValue value = worklist.pop_back_val();
    for (Operand *use : value->getUses()) {
      auto *user = use->getUser();

      if (isUnknownUse(use))
        return false;

      if (isa<CopyValueInst>(user))
        worklist.insert(user);

      // Skip begin_borrow. Consider its end_borrows the use points.
      if (auto *BBI = dyn_cast<BeginBorrowInst>(user)) {
        for (Operand *use : BBI) {
          if (auto *EBI = dyn_cast<EndBorrowInst>(use->getUser()))
            pass.worklist.insert(EBI);
        }
      }
      if (isa<DestroyValueInst>(user))
        continue;

      visitUser(user);
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Rewrite copies and destroys for a single copied definition.
//===----------------------------------------------------------------------===//

static void destroyLastUse(Operand *use, CopyPropagationState &pass) {
  if (isConsuming(use))
    return;

  SILValue srcVal = use->get();
  if (auto *BBI = dyn_cast<BeginBorrowInst>(srcVal))
    srcVal = BBI->getOperand();

  auto *user = use->getUser();
  assert(!isa<TerminatorInst>(user) && "Terminator must consume its operand.");
  SILBuilder B(std::next(user->getIterator()));
  B.setDebugScope(user);
  B.createDestroyValue(user->getLoc(), srcVal);
}

// TODO: Avoid churn. Identify destroys that already complement a last use.
static Invalidation rewriteCopies(SILValue def, CopyPropagationState &pass) {
  SmallSetVector<SILInstruction *, 8> instsToDelete;
  SmallSetVector<SILValue, 8> worklist(def);

  while (!worklist.empty()) {
    SILValue value = worklist.pop_back_val();
    if (auto *copy = dyn_cast<CopyValueInst>(value)) {
      worklist.append(getUserRange(value));
      copy->replaceAllUsesWith(copy->getOperand());
      instsToDelete.insert(copy);
      continue;
    }
    for (Operand *use : value->getUses()) {
      auto *user = use->getUser();
      if (isa<CopyValueInst>(user)) {
        worklist.insert(user);
        continue;
      }
      if (isa<DestroyValueInst>(user)) {
        instsToDelete.insert(user);
        continue;
      }
      destroyLastUse(use, pass);
    }
  }
  recursivelyDeleteTriviallyDeadInstructions(pass.instsToDelete.takeVector(),
                                             /*force=*/true);
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

void CopyPropagation::run() {
  DEBUG(llvm::dbgs() << "*** CopyPropagation: " << F.getName() << "\n");

  assert(getFunction()->hasQualifiedOwnership()
         && "SSA copy propagation is only valid for ownership-SSA");

  SILAnalysis::InvalidationKind invalidation =
    SILAnalysis::InvalidationKind::Nothing;
  
  CopyPropagationState pass(getFunction());
  DenseMap<SILValue> copiedDefs;
  for (auto &BB : *pass.F) {
    for (auto &I : BB) {
      if (auto *copy = dyn_cast<CopyValueInst>(&I))
        copiedDefs.insert(stripCopies(copy));
    }
  }
  for (auto &def : copiedDefs) {
    if (findUsers(def, pass))
      invalidation |= rewriteCopies(def, pass);
  }
  invalidateAnalysis(invalidation);
}

SILTransform *swift::createCopyPropagation() { return new CopyPropagation(); }
