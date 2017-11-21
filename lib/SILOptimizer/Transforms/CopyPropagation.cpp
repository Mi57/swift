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
/// lastUsers  : {SILInstruction}
///
/// 1. Forward walk the instruction stream. Insert the ultimate source of any
///    copy_value into copiedDefs.
///
/// 2. For each copied Def, visit all uses:
///    - Recurse through copies.
///    - Skip over borrows.
///    - Ignore destroys.
///
///    For each use, first walk the use block:
///    - If in liveBlocks and isLiveOut, continue.
///    - If in liveBlocks and !liveout, scan backward from this Use:
///      - If lastUsers.erase(I); lastUsers.insert(Use), stop.
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
/// TODO: This will only be effective for aggregates once SILGen is no longer
/// generating spurious borrows.
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
iterator_range<llvm::mapped_iterator<ValueBase::use_iterator, GetUser>>
getUserRange(SILValue val) {
  return make_range(llvm::map_iterator(val->use_begin(), GetUser()),
                    llvm::map_iterator(val->use_end(), GetUser()));
}

//===----------------------------------------------------------------------===//
// Ownership Abstraction: FIXME: None of this should be in this pass.
//
// (Ownership properties need to be separate from
//  OwnershipCompatibilityUseChecker.
//===----------------------------------------------------------------------===//

/// !!! use apply.getArgumentConvention?
bool doesCallOperConsume(FullApplySite apply, unsigned operIdx) {
  ParameterConvention paramConv;
  if (operIdx == 0)
    paramConv = apply.getSubstCalleeType()->getCalleeConvention();

  unsigned argIndex = apply.getCalleeArgIndex(Op);
  paramConv = apply.getSubstCalleeConv()
                  .getParamInfoForSILArg(argIndex)
                  .getConvention();
  return isConsumedParameter(paramConv);
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
//===----------------------------------------------------------------------===//

/// Mark blocks live in a reverse CFG traversal from this user.
void computeUseBlockLiveness(SILBasicBlock *userBB,
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
/// TODO: This could be expensive for many users within a large block. Consider
/// using the VLA approach of finding live blocks before last users, or just use
/// VLA with a predefined UserSet.
void findLastUser(SILInstruction *user, CopyPropagationState &pass) {
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
void visitUser(SILInstruction *user, CopyPropagationState &pass) {
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
void findUsers(SILValue def, CopyPropagationState &pass) {
  SmallSetVector<SILValue, 8> worklist(def);

  while (!worklist.empty()) {
    SILValue value = worklist.pop_back_val();
    for (Operand *use : value->getUses()) {
      auto *user = use->getUser();

      if (isa<CopyValueInst>(user))
        worklist.insert(user);

      if (auto *borrow = dyn_cast<BeginBorrowInst>(user))
        //!!! user = findEndBorrow(borrow)

      if (isa<DestroyValueInst>(user))
        continue;

      visitUser(user);
    }
  }
}

//===----------------------------------------------------------------------===//
// Rewrite copies and destroys for a single copied definition.
//===----------------------------------------------------------------------===//

void rewriteUser(Operand *use, CopyPropagationState &pass) {
  if (isConsuming(use))
    return;

  // !!! Create the destroy.
}

// TODO: Avoid churn. Identify destroys that already complement a last use.
Invalidation rewriteCopies(SILValue def, CopyPropagationState &pass) {
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
      rewriteUser(use, pass);
    }
  }
  recursivelyDeleteTriviallyDeadInstructions(pass.instsToDelete.takeVector(),
                                             true);
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
    findUsers(def, pass);
    invalidation |= rewriteCopies(def, pass);
  }
  invalidateAnalysis(invalidation);
}

SILTransform *swift::createCopyPropagation() { return new CopyPropagation(); }
