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
/// [WIP] This is meant to complement opaque values. Initially this will run at
/// -O, but eventually may also be adapted to -Onone (as currently designed, it
/// shrinks variable live ranges).
///
/// State:
/// copiedDefs : {SILValue}
/// liveBlocks : {SILBasicBlock, bool isLiveOut}
/// lastUsers  : {SILInstruction}
///
/// 1. Forward walk the instruction stream. Insert the ultimate source of any
/// copy_value into copiedDefs.
///
/// 2. For each copied Def, visit all uses:
///    - Recurse through copies.
///    - Ignore DestroyValue.
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

//===----------------------------------------------------------------------===//
// CopyPropagationState: shared state for the pass's analysis and transforms.
//===----------------------------------------------------------------------===//

namespace {
CopyPropagationState {
  struct LiveBlock {
    PointerIntPair<SILBasicBlock *, 1, bool> bbAndIsLiveOut;

    SILBasicBlock getBB() const { return bbAndIsLiveOut->getPointer(); }
    bool isLiveOut() const { return bbAndIsLiveOut->getInt(); }
  };
  
  SILFunction *F;

  // Per-copied-def state.
  DenseMap<LiveBlock> liveBlocks;
  DenseMap<SILInstruction *> lastUsers;

  CopyPropagationState(SILFunction *F): F(F) {

  void clear() {
    liveBlocks.clear();
    lastUsers.clear();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Eliminate Copies for a single copied definition.
//===----------------------------------------------------------------------===//

/// Update the current def's liveness at the given user.
void visitUser(SILInstruction *user, CopyPropagationState &pass) {
  auto *bb = user->getParent();
  auto &pos = pass.liveBlocks.find(bb);
  if (pos != pass.liveBlocks.end()) {
    if (pos->
    return;

  
}

/// Populate pass.liveBlocks and pass.lastUsers.
void findUsers(SILValue def, CopyPropagationState &pass) {
  SmallVector<SILValue, 8> worklist(def);

  while (!worklist.empty()) {
    SILValue def = worklist.pop_back_val();
    for (Operand *use : def->getUses()) {
      auto *user = use->getUser();
      if (isa<CopyValueInst>(user))
        worklist.push_back(user);

      if (isa<DestroyValueInst>(user))
        continue;

      visitUser(user);
    }
  }
}

Invalidation eliminateCopies(SILValue def, CopyPropagationState &pass) {
  findUsers();
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
  for (auto &def : copiedDefs)
    invalidation |= eliminateCopies(pass);
  
  invalidateAnalysis(invalidation);
}

SILTransform *swift::createCopyPropagation() { return new CopyPropagation(); }
