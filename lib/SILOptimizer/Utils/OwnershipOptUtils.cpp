//===--- OwnershipOptUtils.cpp --------------------------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// Ownership Utilities that rely on SILOptimizer functionality.
///
//===----------------------------------------------------------------------===//

#include "swift/SILOptimizer/Utils/OwnershipOptUtils.h"

#include "swift/Basic/Defer.h"
#include "swift/SIL/BasicBlockUtils.h"
#include "swift/SIL/InstructionUtils.h"
#include "swift/SIL/LinearLifetimeChecker.h"
#include "swift/SIL/MemAccessUtils.h"
#include "swift/SIL/OwnershipUtils.h"
#include "swift/SIL/Projection.h"
#include "swift/SIL/SILArgument.h"
#include "swift/SIL/SILBuilder.h"
#include "swift/SIL/SILInstruction.h"
#include "swift/SILOptimizer/Utils/CFGOptUtils.h"
#include "swift/SILOptimizer/Utils/InstOptUtils.h"
#include "swift/SILOptimizer/Utils/ValueLifetime.h"

using namespace swift;

//===----------------------------------------------------------------------===//
//                          Utility Helper Functions
//===----------------------------------------------------------------------===//

static void cleanupOperandsBeforeDeletion(SILInstruction *oldValue,
                                          InstModCallbacks &callbacks) {
  SILBuilderWithScope builder(oldValue);
  for (auto &op : oldValue->getAllOperands()) {
    if (!op.isLifetimeEnding()) {
      continue;
    }

    switch (op.get().getOwnershipKind()) {
    case OwnershipKind::Any:
      llvm_unreachable("Invalid ownership for value");
    case OwnershipKind::Owned: {
      auto *dvi = builder.createDestroyValue(oldValue->getLoc(), op.get());
      callbacks.createdNewInst(dvi);
      continue;
    }
    case OwnershipKind::Guaranteed: {
      // Should only happen once we model destructures as true reborrows.
      auto *ebi = builder.createEndBorrow(oldValue->getLoc(), op.get());
      callbacks.createdNewInst(ebi);
      continue;
    }
    case OwnershipKind::None:
      continue;
    case OwnershipKind::Unowned:
      llvm_unreachable("Unowned object can never be consumed?!");
    }
    llvm_unreachable("Covered switch isn't covered");
  }
}

static SILPhiArgument *
insertOwnedBaseValueAlongBranchEdge(BranchInst *bi, SILValue innerCopy,
                                    InstModCallbacks &callbacks) {
  auto *destBB = bi->getDestBB();
  // We need to create the phi argument before calling addNewEdgeValueToBranch
  // since it checks that the destination block has enough arguments for the
  // argument.
  auto *phiArg =
      destBB->createPhiArgument(innerCopy->getType(), OwnershipKind::Owned);
  InstructionDeleter deleter(callbacks);
  addNewEdgeValueToBranch(bi, destBB, innerCopy, deleter);

  // Grab our predecessor blocks, ignoring us, add to the branch edge an
  // undef corresponding to our value.
  //
  // We gather all predecessor blocks in a separate array to avoid
  // iterator invalidation issues as we mess with terminators.
  SmallVector<SILBasicBlock *, 8> predecessorBlocks(
      destBB->getPredecessorBlocks());

  for (auto *predBlock : predecessorBlocks) {
    if (predBlock == innerCopy->getParentBlock())
      continue;
    addNewEdgeValueToBranch(
        predBlock->getTerminator(), destBB,
        SILUndef::get(innerCopy->getType(), *destBB->getParent()), deleter);
  }

  return phiArg;
}

//===----------------------------------------------------------------------===//
//                      Ownership RAUW Helper Functions
//===----------------------------------------------------------------------===//

// Determine whether it is valid to replace \p oldValue with \p newValue by
// directly checking ownership requirements. This does not determine whether the
// scope of the newValue can be fully extended.
bool OwnershipRAUWHelper::hasValidRAUWOwnership(SILValue oldValue,
                                                SILValue newValue) {
  auto newOwnershipKind = newValue.getOwnershipKind();

  // If our new kind is ValueOwnershipKind::None, then we are fine. We
  // trivially support that. This check also ensures that we can always
  // replace any value with a ValueOwnershipKind::None value.
  if (newOwnershipKind == OwnershipKind::None)
    return true;

  // If our old ownership kind is ValueOwnershipKind::None and our new kind is
  // not, we may need to do more work that has not been implemented yet. So
  // bail.
  //
  // Due to our requirement that types line up, this can only occur given a
  // non-trivial typed value with None ownership. This can only happen when
  // oldValue is a trivial payloaded or no-payload non-trivially typed
  // enum. That doesn't occur that often so we just bail on it today until we
  // implement this functionality.
  if (oldValue.getOwnershipKind() == OwnershipKind::None)
    return false;

  // First check if oldValue is SILUndef. If it is, then we know that:
  //
  // 1. SILUndef (and thus oldValue) must have OwnershipKind::None.
  // 2. newValue is not OwnershipKind::None due to our check above.
  //
  // Thus we know that we would be replacing a value with OwnershipKind::None
  // with a value with non-None ownership. This is a case we don't support, so
  // we can bail now.
  if (isa<SILUndef>(oldValue))
    return false;

  // Ok, we now know that we do not have SILUndef implying that we must be able
  // to get a module from our value since we must have an argument or an
  // instruction.
  auto *m = oldValue->getModule();
  assert(m);

  // If we are in Raw SIL, just bail at this point. We do not support
  // ownership fixups.
  if (m->getStage() == SILStage::Raw)
    return false;

  return true;
}

// Determine whether it is valid to replace \p oldValue with \p newValue and
// extend the lifetime of \p oldValue to cover the new uses.
//
// This updates the OwnershipFixupContext, populating transitiveBorrowedUses and
// recursiveReborrows.
static bool canFixUpOwnershipForRAUW(SILValue oldValue, SILValue newValue,
                                     OwnershipFixupContext &context) {
  if (!OwnershipRAUWHelper::hasValidRAUWOwnership(oldValue, newValue))
    return false;

  if (oldValue.getOwnershipKind() != OwnershipKind::Guaranteed)
    return true;

  // Check that the old lifetime can be extended and record the necessary
  // book-keeping in the OwnershipFixupContext.
  context.clear();

  if (auto borrowedValue = BorrowedValue(oldValue)) {
    // FIXME!!!: remove this logic and use BorrowedLifetimeExtender.
    SmallPtrSet<SILValue, 4> reborrows;
    auto visitReborrow = [&](Operand *endScope) {
      auto borrowingOper = BorrowingOperand(endScope);
      assert(borrowingOper.isReborrow());
      if (reborrows.insert(
              borrowingOper.getBorrowIntroducingUserResult().value).second) {
        context.recursiveReborrows.push_back(endScope);
      }
    };
    findTransitiveGuaranteedUses(oldValue, context.transitiveBorrowedUses,
                                 visitReborrow);
    for (unsigned idx = 0; idx < context.recursiveReborrows.size(); ++idx) {
      findTransitiveGuaranteedUses(context.recursiveReborrows[idx].getValue(),
                                   context.transitiveBorrowedUses,
                                   visitReborrow);
    }
    return true;
  }
  // Check that an inner guaranteed value is not used by a PointerEscape.
  return findInnerTransitiveGuaranteedUses(oldValue,
                                           context.guaranteedUsePoints);
}

//===----------------------------------------------------------------------===//
//                          BorrowedLifetimeExtender
//===----------------------------------------------------------------------===//

/// Model an extended borrow scope, including transitive reborrows. This applies
/// to "local" borrow scopes (begin_borrow, load_borrow, & phi).
///
/// Allow extending the lifetime of an owned value that dominates this borrowed
/// value across that extended borrow scope. This handles uses of reborrows that
/// are not dominated by the owned value by generating phis and copying the
/// borrowed values the reach this borrow scope from non-dominated paths.
///
/// This produces somewhat canonical owned phis, although that isn't a
/// requirement for valid SIL. Given an owned value, a dominated borrowed value,
/// and a reborrow:
///
///     %ownedValue = ...
///     %borrowedValue = ...
///     %reborrow = phi(%borrowedValue, %otherBorrowedValue)
///
/// %otherBorrowedValue will always be copied even if %ownedValue also dominates
/// %otherBorrowedValue, as such:
///
///     %otherCopy = copy_value %borrowedValue
///     %newPhi = phi(%ownedValue, %otherCopy)
///
/// The immediate effect is to produce an unnecesssary copy, but it avoids
/// extending %ownedValue's liveness to new paths and hopefully simplifies
/// downstream optimization and debugging. Unnecessary copies could be
/// avoided with simple dominance check if it becomes desirable to do so.
struct BorrowedLifetimeExtender {
  BorrowedValue borrowedValue;

  // Owned value currently being extended over borrowedValue.
  SILValue currentOwnedValue;

  InstModCallbacks &callbacks;

  llvm::SmallVector<PhiValue, 4> reborrowedPhis;
  llvm::SmallDenseMap<PhiValue, PhiValue, 4> reborrowedToOwnedPhis;

  /// Check that all reaching operands are handled. This can be removed once the
  /// utility and OSSA representation are stable.
  SWIFT_ASSERT_ONLY_DECL(llvm::SmallDenseSet<PhiOperand, 4> reborrowedOperands);

  /// Initially map the reborrowed phi to an invalid value prior to creating the
  /// owned phi.
  void discoverReborrow(PhiValue reborrowedPhi) {
    if (reborrowedToOwnedPhis.try_emplace(reborrowedPhi, PhiValue()).second) {
      reborrowedPhis.push_back(reborrowedPhi);
    }
  }

  /// Remap the reborrowed phi to an valid owned phi after creating it.
  void mapOwnedPhi(PhiValue reborrowedPhi, PhiValue ownedPhi) {
    reborrowedToOwnedPhis[reborrowedPhi] = ownedPhi;
  }

  /// Get the owned value associated with this reborrowed operand, or return an
  /// invalid SILValue indicating that the borrowed lifetime does not reach this
  /// operand.
  SILValue getExtendedOwnedValue(PhiOperand reborrowedOper) {
    // If this operand reborrows the original borrow, then the currentOwned phi
    // reaches it directly.
    SILValue borrowSource = reborrowedOper.getSource();
    if (borrowSource == borrowedValue.value)
      return currentOwnedValue;

    // Check if the borrowed operand's source is already mapped to an owned phi.
    auto reborrowedAndOwnedPhi = reborrowedToOwnedPhis.find(borrowSource);
    if (reborrowedAndOwnedPhi != reborrowedToOwnedPhis.end()) {
      // Return the already-mapped owned phi.
      assert(reborrowedOperands.erase(reborrowedOper));
      return reborrowedAndOwnedPhi->second;
    }
    // The owned value does not reach this reborrowed operand.
    assert(
        !reborrowedOperands.count(reborrowedOper)
        && "reachable borrowed phi operand must be mapped to an owned value");
    return SILValue();
  }

public:
  /// Precondition: \p borrowedValue must introduce a local borrow scope
  /// (begin_borrow, load_borrow, & phi).
  BorrowedLifetimeExtender(BorrowedValue borrowedValue,
                           InstModCallbacks &callbacks)
      : borrowedValue(borrowedValue), callbacks(callbacks) {
    assert(borrowedValue.isLocalScope() && "expect a valid borrowed value");
  }

  /// Extend \p ownedValue over this extended borrow scope.
  ///
  /// Precondition: \p ownedValue dominates this borrowed value.
  void extendOverBorrowScopeAndConsume(SILValue ownedValue);

protected:
  void analyzeExtendedScope();

  SILValue createCopyAtEdge(PhiOperand reborrowOper);

  void destroyAtScopeEnd(SILValue ownedValue, BorrowedValue pairedBorrow);
};

// Gather all transitive phi-reborrows and check that all the borrowed uses can
// be found with no escapes.
//
// Calls discoverReborrow to populate reborrowedPhis.
void BorrowedLifetimeExtender::analyzeExtendedScope() {
  auto visitReborrow = [&](Operand *endScope) {
    if (auto borrowingOper = BorrowingOperand(endScope)) {
      assert(borrowingOper.isReborrow());

      SWIFT_ASSERT_ONLY(reborrowedOperands.insert(endScope));

      // TODO: if non-phi reborrows are added, handle multiple results.
      discoverReborrow(borrowingOper.getBorrowIntroducingUserResult().value);
    }
    return true;
  };

  bool result = borrowedValue.visitLocalScopeEndingUses(visitReborrow);
  assert(result && "visitReborrow always succeeds, escapes are irrelevant");

  // Note: Iterate in the same manner as findExtendedTransitiveGuaranteedUses(),
  // but using BorrowedLifetimeExtender's own reborrowedPhis.
  for (unsigned idx = 0; idx < reborrowedPhis.size(); ++idx) {
    auto borrowedValue = BorrowedValue(reborrowedPhis[idx]);
    result = borrowedValue.visitLocalScopeEndingUses(visitReborrow);
    assert(result && "visitReborrow always succeeds, escapes are irrelevant");
  }
}

// Insert a copy on this edge. This might not be necessary if the owned
// value dominates this path, but this avoids forcing the owned value to be
// live across new paths.
//
// TODO: consider copying the base of the borrowed value instead of the
// borrowed value directly. It's likely that the copy is used outside of the
// borrow scope, in which case, canonicalizeOSSA will create a copy outside
// the borrow scope anyway. However, we can't be sure that the base is the
// same type.
//
// TODO: consider reusing copies that dominate multiple reborrowed
// operands. Howeer, this requires copying in an earlier block and inserting
// post-dominating destroys, which may be better handled in an ownership phi
// canonicalization pass.
SILValue BorrowedLifetimeExtender::createCopyAtEdge(PhiOperand reborrowOper) {
  auto *branch = reborrowOper.getBranch();
  auto loc = RegularLocation::getAutoGeneratedLocation(branch->getLoc());
  auto *copy = SILBuilderWithScope(branch).createCopyValue(
      loc, reborrowOper.getSource());
  callbacks.createdNewInst(copy);
  return copy;
}

// Destroy \p ownedValue at \p pairedBorrow's scope-ending uses, excluding
// reborrows.
//
// Precondition: ownedValue takes ownership of its value at the same point as
// pairedBorrow. e.g. an owned and guaranteed pair of phis.
void BorrowedLifetimeExtender::destroyAtScopeEnd(SILValue ownedValue,
                                                 BorrowedValue pairedBorrow) {
  pairedBorrow.visitLocalScopeEndingUses([&](Operand *scopeEnd) {
    if (scopeEnd->getOperandOwnership() == OperandOwnership::Reborrow)
      return true;

    auto *endInst = scopeEnd->getUser();
    assert(!isa<TermInst>(endInst) && "branch must be a reborrow");
    auto *destroyPt = &*std::next(endInst->getIterator());
    auto *destroy = SILBuilderWithScope(destroyPt).createDestroyValue(
        destroyPt->getLoc(), ownedValue);
    callbacks.createdNewInst(destroy);
    return true;
  });
}

// Insert and map an owned phi for each reborrowed phi.
//
// For each reborrowed phi, insert a copy on each edge that does not originate
// from the extended borrowedValue.
//
// TODO: If non-phi reborrows are added, they would also need to be
// mapped to their owned counterpart. This means generating new owned
// struct/destructure instructions.
void BorrowedLifetimeExtender::
extendOverBorrowScopeAndConsume(SILValue ownedValue) {
  currentOwnedValue = ownedValue;

  // Populate the reborrowedPhis vector.
  analyzeExtendedScope();

  InstructionDeleter deleter(callbacks);

  // Generate and map the phis with undef operands first, in case of recursion.
  auto undef = SILUndef::get(ownedValue->getType(), *ownedValue->getFunction());
  for (PhiValue reborrowedPhi : reborrowedPhis) {
    auto *phiBlock = reborrowedPhi.phiBlock;
    auto *ownedPhi = phiBlock->createPhiArgument(ownedValue->getType(),
                                                 OwnershipKind::Owned);
    for (auto *predBlock : phiBlock->getPredecessorBlocks()) {
      TermInst *ti = predBlock->getTerminator();
      addNewEdgeValueToBranch(ti, phiBlock, undef, deleter);
    }
    mapOwnedPhi(reborrowedPhi, PhiValue(ownedPhi));
  }
  // Generate copies and set the phi operands.
  for (PhiValue reborrowedPhi : reborrowedPhis) {
    PhiValue ownedPhi = reborrowedToOwnedPhis[reborrowedPhi];
    reborrowedPhi.getValue()->visitIncomingPhiOperands(
        // For each reborrowed operand, get the owned value for that edge,
        // and set the owned phi's operand.
        [&](Operand *reborrowedOper) {
          SILValue ownedVal = getExtendedOwnedValue(reborrowedOper);
          if (!ownedVal) {
            ownedVal = createCopyAtEdge(reborrowedOper);
          }
          BranchInst *branch = PhiOperand(reborrowedOper).getBranch();
          branch->getOperandRef(ownedPhi.argIndex).set(ownedVal);
          return true;
        });
  }
  assert(reborrowedOperands.empty() && "not all phi operands are handled");

  // Create destroys at the last uses.
  destroyAtScopeEnd(ownedValue, borrowedValue);
  for (PhiValue reborrowedPhi : reborrowedPhis) {
    PhiValue ownedPhi = reborrowedToOwnedPhis[reborrowedPhi];
    destroyAtScopeEnd(ownedPhi, BorrowedValue(reborrowedPhi));
  }
}

//===----------------------------------------------------------------------===//
//                        Ownership Lifetime Extender
//===----------------------------------------------------------------------===//

namespace {

struct OwnershipLifetimeExtender {
  OwnershipFixupContext &ctx;

  /// Create a new copy of \p value assuming that our caller will clean up the
  /// copy along all paths that go through consuming point. Operationally this
  /// means that the API will insert compensating destroy_value on the copy
  /// along all paths that do not go through consuming point.
  ///
  /// DISCUSSION: If \p consumingPoint is an instruction that forwards \p value,
  /// calling this and then RAUWing with \p value guarantee that \p value will
  /// be consumed by the forwarding instruction's results consuming uses.
  CopyValueInst *createPlusOneCopy(SILValue value,
                                   SILInstruction *consumingPoint);

  /// Create a copy of \p value that covers all of \p range and insert all
  /// needed destroy_values. We assume that no uses in \p range consume \p
  /// value.
  CopyValueInst *createPlusZeroCopy(SILValue value, ArrayRef<Operand *> range) {
    return createPlusZeroCopy<ArrayRef<Operand *>>(value, range);
  }

  /// Create a copy of \p value that covers all of \p range and insert all
  /// needed destroy_values. We assume that all uses in \p range do not consume
  /// \p value.
  ///
  /// We return our copy_value to the user at +0 to show that they do not need
  /// to insert cleanup destroys.
  template <typename RangeTy>
  CopyValueInst *createPlusZeroCopy(SILValue value, const RangeTy &range);

  /// Borrow \p newValue over the extended lifetime of \p borrowedValue.
  BeginBorrowInst *borrowCopyOverScope(SILValue newValue,
                                       BorrowedValue borrowedValue);

  /// Borrow-copy \p newValue over \p guaranteedUses. Copy newValue, borrow the
  /// copy, and extend the lifetime of the borrow-copy over guaranteedUsePoints.
  ///
  /// \p borrowPoint is a value whose definition will be the location of
  /// the new borrow.
  template <typename RangeTy>
  BeginBorrowInst *
  borrowCopyOverGuaranteedUses(SILValue newValue,
                               SILBasicBlock::iterator borrowPoint,
                               RangeTy guaranteedUsePoints);

  /// Borrow \p newValue over the lifetime of \p guaranteedValue. Return the
  /// new guaranteed value.
  SILValue borrowOverValue(SILValue newValue, SILValue guaranteedValue);

  /// Borrow \p newValue over \p singleGuaranteedUse. Return the
  /// new guaranteed value.
  ///
  /// Precondition: if \p use ends a borrow scope, then \p newValue dominates
  /// the BorrowedValue that begins the scope.
  SILValue borrowOverSingleUse(SILValue newValue,
                               Operand *singleGuaranteedUse);

  // --- FIXME!!! remove the following (replaced by BorrowedLifetimeExtender)

  /// Create a new borrow scope for \p newValue that is cleaned up along all
  /// paths that do not go through consuming point. The caller is expected to
  /// consumg \p newValue at \p consumingPoint since we insert a destroy_value
  /// right after wards.
  BeginBorrowInst *originalCreatePlusOneBorrow(SILValue newValue,
                                               SILInstruction *consumingPoint);
};

} // end anonymous namespace

/// Lifetime extend \p value over \p consumingPoint, assuming that \p
/// consumingPoint will consume \p value after the client performs replacement
/// (this implicit destruction on the caller-side makes it a "plus-one"
/// copy). Destroy \p copy on all paths that don't reach \p consumingPoint.
///
/// Precondition: \p value is owned
///
/// Precondition: \p consumingPoint is dominated by \p value
CopyValueInst *
OwnershipLifetimeExtender::createPlusOneCopy(SILValue value,
                                             SILInstruction *consumingPoint) {
  auto *copyPoint = value->getNextInstruction();
  auto loc = copyPoint->getLoc();
  auto *copy = SILBuilderWithScope(copyPoint).createCopyValue(loc, value);

  auto &callbacks = ctx.callbacks;
  callbacks.createdNewInst(copy);

  auto *result = copy;
  findJointPostDominatingSet(
      copy->getParent(), consumingPoint->getParent(),
      // inputBlocksFoundDuringWalk.
      [&](SILBasicBlock *loopBlock) {
        // Create an extra copy when the consuming point is inside a
        // loop and both copyPoint and the destroy points are outside the
        // loop. This copy will be consumed in the same block. The original
        // value will be destroyed on all paths exiting the loop.
        //
        // Since copyPoint dominates consumingPoint, it must be outside the
        // loop. Otherwise backward traversal would have stopped at copyPoint
        assert(loopBlock == consumingPoint->getParent());
        auto front = loopBlock->begin();
        SILBuilderWithScope newBuilder(front);
        result = newBuilder.createCopyValue(front->getLoc(), copy);
        callbacks.createdNewInst(result);
      },
      // Leaky blocks that never reach consumingPoint.
      [&](SILBasicBlock *postDomBlock) {
        auto front = postDomBlock->begin();
        SILBuilderWithScope newBuilder(front);
        auto *dvi = newBuilder.createDestroyValue(front->getLoc(), copy);
        callbacks.createdNewInst(dvi);
      });
  return result;
}

// A copy_value that we lifetime extend with destroy_value over range. We assume
// all instructions passed into range do not consume value.
template <typename RangeTy>
CopyValueInst *
OwnershipLifetimeExtender::createPlusZeroCopy(SILValue value,
                                              const RangeTy &range) {
  auto *newValInsertPt = value->getDefiningInsertionPoint();
  assert(newValInsertPt);

  CopyValueInst *copy;

  if (!isa<SILArgument>(value)) {
    SILBuilderWithScope::insertAfter(newValInsertPt, [&](SILBuilder &builder) {
      copy = builder.createCopyValue(builder.getInsertionPointLoc(), value);
    });
  } else {
    SILBuilderWithScope builder(newValInsertPt);
    copy = builder.createCopyValue(newValInsertPt->getLoc(), value);
  }

  auto &callbacks = ctx.callbacks;
  callbacks.createdNewInst(copy);

  auto opRange = makeUserRange(range);
  ValueLifetimeAnalysis lifetimeAnalysis(copy, opRange);
  ValueLifetimeAnalysis::Frontier frontier;
  bool result = lifetimeAnalysis.computeFrontier(
      frontier, ValueLifetimeAnalysis::DontModifyCFG, &ctx.deBlocks);
  assert(result);

  while (!frontier.empty()) {
    auto *insertPt = frontier.pop_back_val();
    SILBuilderWithScope frontierBuilder(insertPt);
    auto *dvi = frontierBuilder.createDestroyValue(insertPt->getLoc(), copy);
    callbacks.createdNewInst(dvi);
  }

  return copy;
}

/// Borrow \p newValue over the extended lifetime of \p borrowedValue.
///
/// Precondition: \p newValue dominates borrowedValue.
BeginBorrowInst *
OwnershipLifetimeExtender::borrowCopyOverScope(SILValue newValue,
                                               BorrowedValue borrowedValue) {
  assert(borrowedValue.isLocalScope() && "SILFunctionArg is already handled");

  SILInstruction *borrowPoint = borrowedValue.value->getNextInstruction();
  auto loc = RegularLocation::getAutoGeneratedLocation(borrowPoint->getLoc());
  SILBuilderWithScope builder(borrowPoint);
  auto *copy = builder.createCopyValue(loc, newValue);
  ctx.callbacks.createdNewInst(copy);

  // Extend the new copy's lifetime over borrowedValue's scope and destroy it on
  // all paths through borrowedValue. Since copy is in the same block as
  // borrowedValue, no extra destroys are needed.
  BorrowedLifetimeExtender(borrowedValue, ctx.callbacks)
      .extendOverBorrowScopeAndConsume(copy);

  auto *borrow = builder.createBeginBorrow(loc, copy);
  ctx.callbacks.createdNewInst(borrow);
  return borrow;
}

/// Borrow-copy \p newValue over \p guaranteedUses. Copy newValue, borrow the
/// copy, and extend the lifetime of the borrow-copy over guaranteedUses.
///
/// \p borrowPoint is a the insertion point of the new borrow.
///
/// Precondition: \p newValue dominates \p borrowPoint which dominates \p
/// guaranteedUses
///
/// Precondition: None of \p guaranteedUses are lifetime ending.
template <typename RangeTy>
BeginBorrowInst *OwnershipLifetimeExtender::borrowCopyOverGuaranteedUses(
    SILValue newValue, SILBasicBlock::iterator borrowPoint,
    RangeTy guaranteedUsePoints) {

  auto loc = RegularLocation::getAutoGeneratedLocation(borrowPoint->getLoc());
  SILBuilderWithScope builder(borrowPoint);
  auto *copy = builder.createCopyValue(loc, newValue);
  auto *borrow = builder.createBeginBorrow(loc, copy);
  ctx.callbacks.createdNewInst(copy);
  ctx.callbacks.createdNewInst(borrow);

  // We don't expect an empty guaranteedUsePoints. If it happens, then the newly
  // created copy will never be destroyed.
  assert(!guaranteedUsePoints.empty());
  auto opRange = makeUserRange(guaranteedUsePoints);
  ValueLifetimeAnalysis lifetimeAnalysis(copy, opRange);
  ValueLifetimeAnalysis::Frontier frontier;
  bool result = lifetimeAnalysis.computeFrontier(
      frontier, ValueLifetimeAnalysis::DontModifyCFG, &ctx.deBlocks);
  assert(result);

  auto &callbacks = ctx.callbacks;
  while (!frontier.empty()) {
    auto *insertPt = frontier.pop_back_val();
    SILBuilderWithScope frontierBuilder(insertPt);
    // Use an auto-generated location here, because insertPt may have an
    // incompatible LocationKind
    auto loc = RegularLocation::getAutoGeneratedLocation(insertPt->getLoc());
    auto *endBorrow = frontierBuilder.createEndBorrow(loc, borrow);
    auto *destroy = frontierBuilder.createDestroyValue(loc, copy);
    callbacks.createdNewInst(endBorrow);
    callbacks.createdNewInst(destroy);
  }
  return borrow;
}

// Return the borrow position when replacing guaranteedValue with newValue.
//
// Precondition: newValue's block dominates and reaches guaranteedValue's block.
//
// Postcondition: The returned instruction's block is guaranteedValue's block.
//
// If \p newValue and \p guaranteedValue are in the same block, borrow at the
// newValue just in case it is defined later in the block (to avoid scanning
// instructions). Otherwise, borrow in the guaranteedValue's block to avoid
// introducing the borrow scope too early--not only would this require extra
// cleanup, but it would hinder optimization.
static SILBasicBlock::iterator getBorrowPoint(SILValue newValue,
                                              SILValue guaranteedValue) {
  if (newValue->getParentBlock() == guaranteedValue->getParentBlock())
    return newValue->getNextInstruction()->getIterator();

  return guaranteedValue->getNextInstruction()->getIterator();
}

/// Borrow \p newValue over the lifetime of \p guaranteedValue. Return the
/// new guaranteed value.
///
/// FIXME: Consider replacing all of newValue's uses with the new copy of
/// newValue. This may allow newValue's original borrow scope to be removed,
/// which then allows the copy to be removed. The result would be a single
/// borrow scope over all newValue's and guaranteedValue's uses, which is
/// usually preferrable to a new copy and separate borrow scope. When doing
/// this, we can use newValue as the borrow point instead of getBorrowPoint.
SILValue
OwnershipLifetimeExtender::borrowOverValue(SILValue newValue,
                                           SILValue guaranteedValue) {
  // Avoid borrowing guaranteed function arguments.
  if (isa<SILFunctionArgument>(newValue)
      && newValue.getOwnershipKind() == OwnershipKind::Guaranteed) {
    return newValue;
  }
  auto borrowedValue = BorrowedValue(guaranteedValue);
  if (borrowedValue.isLocalScope()) {
    return borrowCopyOverScope(newValue, borrowedValue);
  }
  auto borrowPt = getBorrowPoint(newValue, guaranteedValue);
  return borrowCopyOverGuaranteedUses(newValue, borrowPt,
                                      ArrayRef<Operand *>(ctx.guaranteedUsePoints));
}

// Borrow \p newValue over \p singleGuaranteedUse. Return the new guaranteed
// value.
//
// Precondition: \p newValue dominates dominates \p singleGuaranteedUse.
//
// Precondition: If \p singleGuaranteedUse ends a borrowed lifetime, the \p
// newValue also dominates the beginning of the borrow scope.
//
// If \p singleGuaranteedUse is lifetime-ending, then two forms
// of cleanup are performed, anticipating that singleGuaranteedUse will be
// replaced with the returned value.
//
// 1. Insert an end_borrow for the original borrow at the point of the replaced
// use.
//
// 2. Insert end_borrows for the new borrow at all the original borrow's
// scope-ending uses that aren't being replaced.
SILValue
OwnershipLifetimeExtender::borrowOverSingleUse(SILValue newValue,
                                               Operand *singleGuaranteedUse) {
  // Avoid borrowing guaranteed function arguments.
  if (isa<SILFunctionArgument>(newValue)
      && newValue.getOwnershipKind() == OwnershipKind::Guaranteed) {
    return newValue;
  }
  if (!singleGuaranteedUse->isLifetimeEnding()) {
    auto borrowPt = newValue->getNextInstruction()->getIterator();
    return borrowCopyOverGuaranteedUses(
        newValue, borrowPt, ArrayRef<Operand *>(singleGuaranteedUse));
  }
  // A guaranteed lifetime-ending use is always defined by a BorrowedValue.
  auto oldBorrowedVal = BorrowedValue(singleGuaranteedUse->get());
  BeginBorrowInst *newBeginBorrow =
      borrowCopyOverScope(newValue, oldBorrowedVal);

  // Cleanup the original scope, anticipating that it will lose an end-point.
  SILInstruction *usePoint = singleGuaranteedUse->getUser();
  auto *endOldBorrow = SILBuilderWithScope(usePoint).createEndBorrow(
      usePoint->getLoc(), oldBorrowedVal.value);
  ctx.callbacks.createdNewInst(endOldBorrow);

  // Cleanup the new scope since it only inherits one end-point.
  oldBorrowedVal.visitLocalScopeEndingUses([&](Operand *endScope) {
    auto borrowingOper = BorrowingOperand(endScope);
    if (borrowingOper.isReborrow())
      return true;

    auto *oldEndBorrow = endScope->getUser();
    auto *endNewBorrow =
        SILBuilderWithScope(oldEndBorrow)
            .createEndBorrow(oldEndBorrow->getLoc(), newBeginBorrow);
    ctx.callbacks.createdNewInst(endNewBorrow);
    return true;
  });
  return newBeginBorrow;
}

// TODO: replace with borrowOverValue/borrowOverSingleUse
BeginBorrowInst *OwnershipLifetimeExtender::originalCreatePlusOneBorrow(
    SILValue value, SILInstruction *consumingPoint) {
  auto *newValInsertPt = value->getDefiningInsertionPoint();
  assert(newValInsertPt);
  CopyValueInst *copy;
  BeginBorrowInst *borrow;
  if (!isa<SILArgument>(value)) {
    SILBuilderWithScope::insertAfter(newValInsertPt, [&](SILBuilder &builder) {
      copy = builder.createCopyValue(builder.getInsertionPointLoc(), value);
      borrow = builder.createBeginBorrow(builder.getInsertionPointLoc(), copy);
    });
  } else {
    SILBuilderWithScope builder(newValInsertPt);
    copy = builder.createCopyValue(newValInsertPt->getLoc(), value);
    borrow = builder.createBeginBorrow(newValInsertPt->getLoc(), copy);
  }

  auto &callbacks = ctx.callbacks;
  callbacks.createdNewInst(copy);
  callbacks.createdNewInst(borrow);

  auto *result = borrow;
  findJointPostDominatingSet(
      newValInsertPt->getParent(), consumingPoint->getParent(),
      // inputBlocksFoundDuringWalk.
      [&](SILBasicBlock *loopBlock) {
        // This must be consumingPoint->getParent() since we only have one
        // consuming use. In this case, we know that this is the consuming
        // point where we will need a control equivalent copy_value (and that
        // destroy_value will be put for the out of loop value as appropriate.
        assert(loopBlock == consumingPoint->getParent());
        auto front = loopBlock->begin();
        SILBuilderWithScope newBuilder(front);
        result = newBuilder.createBeginBorrow(front->getLoc(), borrow);
        callbacks.createdNewInst(result);

        llvm_unreachable("Should never visit this!");
      },
      // Input blocks in joint post dom set. We don't care about thse.
      [&](SILBasicBlock *postDomBlock) {
        auto front = postDomBlock->begin();
        SILBuilderWithScope newBuilder(front);
        auto *ebi = newBuilder.createEndBorrow(front->getLoc(), borrow);
        callbacks.createdNewInst(ebi);
        auto *dvi = newBuilder.createDestroyValue(front->getLoc(), copy);
        callbacks.createdNewInst(dvi);
      });
  return result;
}

//===----------------------------------------------------------------------===//
//                            Reborrow Elimination
//===----------------------------------------------------------------------===//

static void eliminateReborrowsOfRecursiveBorrows(
    ArrayRef<PhiOperand> transitiveReborrows,
    SmallVectorImpl<Operand *> &usePoints, InstModCallbacks &callbacks) {
  SmallVector<std::pair<SILPhiArgument *, SILPhiArgument *>, 8>
      baseBorrowedValuePair;
  // Ok, we have transitive reborrows.
  for (auto it : transitiveReborrows) {
    // We eliminate the reborrow by creating a new copy+borrow at the reborrow
    // edge from the base value and using that for the reborrow instead of the
    // actual value. We of course insert an end_borrow for our original incoming
    // value.
    auto *bi = cast<BranchInst>(it.predBlock->getTerminator());
    auto &op = bi->getOperandRef(it.argIndex);
    BorrowingOperand borrowingOperand(&op);
    SILValue value = borrowingOperand->get();
    SILBuilderWithScope reborrowBuilder(bi);
    // Use an auto-generated location here, because the branch may have an
    // incompatible LocationKind
    auto loc = RegularLocation::getAutoGeneratedLocation(bi->getLoc());
    auto *innerCopy = reborrowBuilder.createCopyValue(loc, value);
    auto *innerBorrow = reborrowBuilder.createBeginBorrow(loc, innerCopy);
    auto *outerEndBorrow = reborrowBuilder.createEndBorrow(loc, value);

    callbacks.createdNewInst(innerCopy);
    callbacks.createdNewInst(innerBorrow);
    callbacks.createdNewInst(outerEndBorrow);

    // Then set our borrowing operand to take our innerBorrow instead of value
    // (whose lifetime we just ended).
    callbacks.setUseValue(*borrowingOperand, innerBorrow);
    // Add our outer end borrow as a use point to make sure that we extend our
    // base value to this point.
    usePoints.push_back(&outerEndBorrow->getAllOperands()[0]);

    // Then check if in our destination block, we have further reborrows. If we
    // do, we need to recursively process them.
    auto *borrowedArg =
        const_cast<SILPhiArgument *>(bi->getArgForOperand(*borrowingOperand));
    auto *baseArg =
        insertOwnedBaseValueAlongBranchEdge(bi, innerCopy, callbacks);
    baseBorrowedValuePair.emplace_back(baseArg, borrowedArg);
  }

  // Now recursively update all further reborrows...
  while (!baseBorrowedValuePair.empty()) {
    SILPhiArgument *baseArg;
    SILPhiArgument *borrowedArg;
    std::tie(baseArg, borrowedArg) = baseBorrowedValuePair.pop_back_val();

    for (auto *use : borrowedArg->getConsumingUses()) {
      // If our consuming use is an end of scope marker, we need to end
      // the lifetime of our base arg.
      if (isEndOfScopeMarker(use->getUser())) {
        SILBuilderWithScope::insertAfter(use->getUser(), [&](SILBuilder &b) {
          auto *dvi = b.createDestroyValue(b.getInsertionPointLoc(), baseArg);
          callbacks.createdNewInst(dvi);
        });
        continue;
      }

      // Otherwise, we have a reborrow. For now our reborrows must be
      // phis. Add our owned value as a new argument of that phi along our
      // edge and undef along all other edges.
      auto borrowingOp = BorrowingOperand(use);
      auto *brInst = cast<BranchInst>(borrowingOp.op->getUser());
      auto *newBorrowedPhi = brInst->getArgForOperand(*borrowingOp);
      auto *newBasePhi =
          insertOwnedBaseValueAlongBranchEdge(brInst, baseArg, callbacks);
      baseBorrowedValuePair.emplace_back(newBasePhi, newBorrowedPhi);
    }
  }
}

//===----------------------------------------------------------------------===//
//                OwnershipRAUWPrepare - RAUW + fix ownership
//===----------------------------------------------------------------------===//

/// Given an old value and a new value, lifetime extend new value as appropriate
/// so we can RAUW new value with old value and preserve ownership
/// invariants. We leave fixing up the lifetime of old value to our caller.
namespace {

struct OwnershipRAUWPrepare {
  SILValue oldValue;
  OwnershipFixupContext &ctx;

  OwnershipLifetimeExtender getLifetimeExtender() { return {ctx}; }

  const InstModCallbacks &getCallbacks() const { return ctx.callbacks; }

  // For terminator results, the consuming point is the predecessor's
  // terminator. This avoids destroys on unused paths. It is also the
  // instruction which will be deleted, thus needs operand cleanup.
  SILInstruction *getConsumingPoint() const {
    if (auto *blockArg = dyn_cast<SILPhiArgument>(oldValue))
      return blockArg->getTerminatorForResult();

    return cast<SingleValueInstruction>(oldValue);
  }

  SILValue prepareReplacement(SILValue newValue);

private:
  SILValue prepareUnowned(SILValue newValue);
};

} // anonymous namespace

SILValue OwnershipRAUWPrepare::prepareUnowned(SILValue newValue) {
  auto &callbacks = ctx.callbacks;
  switch (newValue.getOwnershipKind()) {
  case OwnershipKind::None:
    llvm_unreachable("Should have been handled elsewhere");
  case OwnershipKind::Any:
    llvm_unreachable("Invalid for values");
  case OwnershipKind::Unowned:
    // An unowned value can always be RAUWed with another unowned value.
    return newValue;
  case OwnershipKind::Guaranteed: {
    // If we have an unowned value that we want to replace with a guaranteed
    // value, we need to ensure that the guaranteed value is live at all use
    // points of the unowned value. If so, just replace and continue.
    //
    // TODO: Implement this for more interesting cases.
    if (isa<SILFunctionArgument>(newValue))
      return newValue;

    // Otherwise, we need to lifetime extend the borrow over all of the use
    // points. To do so, we copy the value, borrow it, and insert an unchecked
    // ownership conversion to unowned at all uses that are terminator uses.
    //
    // We need to insert the conversion since if we have a non-argument
    // guaranteed value since its scope will end before the terminator so we
    // need to convert the value to unowned early.
    //
    // TODO: Do we need a separate array here?
    SmallVector<Operand *, 8> oldValueUses(oldValue->getUses());
    for (auto *use : oldValueUses) {
      if (auto *ti = dyn_cast<TermInst>(use->getUser())) {
        if (ti->isFunctionExiting()) {
          SILBuilderWithScope builder(ti);
          auto *newInst = builder.createUncheckedOwnershipConversion(
              ti->getLoc(), use->get(), OwnershipKind::Unowned);
          callbacks.createdNewInst(newInst);
          callbacks.setUseValue(use, newInst);
        }
      }
    }

    auto extender = getLifetimeExtender();
    auto borrowPt = getBorrowPoint(newValue, oldValue);
    SILValue borrow = extender.borrowCopyOverGuaranteedUses(
        newValue, borrowPt, oldValue->getUses());
    return borrow;
  }
  case OwnershipKind::Owned: {
    // If we have an unowned value that we want to replace with an owned value,
    // we first check if the owned value is live over all use points of the old
    // value. If so, just RAUW and continue.
    //
    // TODO: Implement this.

    // Otherwise, insert a copy of the owned value and lifetime extend that over
    // all uses of the value and then RAUW.
    //
    // NOTE: For terminator uses, we funnel the use through an
    // unchecked_ownership_conversion to ensure that we can end the lifetime of
    // our owned/guaranteed value before the terminator.
    SmallVector<Operand *, 8> oldValueUses(oldValue->getUses());
    for (auto *use : oldValueUses) {
      if (auto *ti = dyn_cast<TermInst>(use->getUser())) {
        if (ti->isFunctionExiting()) {
          SILBuilderWithScope builder(ti);
          auto *newInst = builder.createUncheckedOwnershipConversion(
              ti->getLoc(), use->get(), OwnershipKind::Unowned);
          callbacks.createdNewInst(newInst);
          callbacks.setUseValue(use, newInst);
        }
      }
    }
    auto extender = getLifetimeExtender();
    SILValue copy = extender.createPlusZeroCopy(newValue, oldValue->getUses());
    return copy;
  }
  }
  llvm_unreachable("covered switch isn't covered?!");
}

SILValue OwnershipRAUWPrepare::prepareReplacement(SILValue newValue) {
  assert(oldValue->getFunction()->hasOwnership());
  assert(OwnershipRAUWHelper::hasValidRAUWOwnership(oldValue, newValue) &&
      "Should have checked if can perform this operation before calling it?!");
  // If our new value is just none, we can pass anything to do it so just RAUW
  // and return.
  //
  // NOTE: This handles RAUWing with undef.
  if (newValue.getOwnershipKind() == OwnershipKind::None)
    return newValue;
  
  assert(oldValue.getOwnershipKind() != OwnershipKind::None);

  switch (oldValue.getOwnershipKind()) {
  case OwnershipKind::None:
    // If our old value was none and our new value is not, we need to do
    // something more complex that we do not support yet, so bail. We should
    // have not called this function in such a case.
    llvm_unreachable("Should have been handled elsewhere");
  case OwnershipKind::Any:
    llvm_unreachable("Invalid for values");
  case OwnershipKind::Guaranteed: {
    return getLifetimeExtender().borrowOverValue(newValue, oldValue);
  }
  case OwnershipKind::Owned: {
    // If we have an owned value that we want to replace with a value with any
    // other non-None ownership, we need to copy the other value for a
    // lifetimeEnding RAUW, RAUW the value, and insert a destroy_value of
    // the original value.
    auto extender = getLifetimeExtender();
    auto *consumingPoint = getConsumingPoint();
    SILValue copy = extender.createPlusOneCopy(newValue, consumingPoint);
    cleanupOperandsBeforeDeletion(consumingPoint, ctx.callbacks);
    return copy;
  }
  case OwnershipKind::Unowned: {
    return prepareUnowned(newValue);
  }
  }
  llvm_unreachable("Covered switch isn't covered?!");
}

//===----------------------------------------------------------------------===//
//                     Interior Pointer Operand Rebasing
//===----------------------------------------------------------------------===//

/// Return an address equivalent to \p newValue that can be used to replace all
/// uses of \p oldValue.
///
/// Precondition: RAUW of two addresses
SILValue
OwnershipRAUWHelper::getReplacementAddress() {
  assert(oldValue->getType().isAddress() && newValue->getType().isAddress());

  // If newValue was not generated by an interior pointer, then it cannot
  // be within a borrow scope, so direct replacement works.
  if (!requiresCopyBorrowAndClone())
    return newValue;

  // newValue may be within a borrow scope, and oldValue may have uses that are
  // outside of newValue's borrow scope.
  //
  // So, we need to copy/borrow the base value of the interior pointer to
  // lifetime extend the base value over the new uses. Then we clone the
  // interior pointer instruction and change the clone to use our new borrowed
  // value. Then we RAUW as appropriate.
  OwnershipLifetimeExtender extender{*ctx};
  auto &extraInfo = ctx->extraAddressFixupInfo;
  auto intPtr = *extraInfo.intPtrOp;
  auto borrowPt = getBorrowPoint(newValue, oldValue);
  BeginBorrowInst *bbi = extender.borrowCopyOverGuaranteedUses(
      intPtr->get(), borrowPt,
      llvm::makeArrayRef(extraInfo.allAddressUsesFromOldValue));
  auto bbiNext = &*std::next(bbi->getIterator());
  auto *newIntPtrUser =
      cast<SingleValueInstruction>(intPtr->getUser()->clone(bbiNext));
  ctx->callbacks.createdNewInst(newIntPtrUser);
  newIntPtrUser->setOperand(0, bbi);

  // Now that we have extended our lifetime as appropriate, we need to recreate
  // the access path from newValue to intPtr but upon newIntPtr. Then we make it
  // use newIntPtr.
  auto *intPtrUser = cast<SingleValueInstruction>(intPtr->getUser());

  // This cloner invocation must match the canCloneUseDefChain check in the
  // constructor.
  auto checkBase = [&](SILValue srcAddr) {
    return (srcAddr == intPtrUser) ? SILValue(newIntPtrUser) : SILValue();
  };
  SILValue clonedAddr =
    cloneUseDefChain(newValue, oldValue->getDefiningInsertionPoint(),
                     checkBase);
  assert(clonedAddr != newValue && "expect at least the base to be replaced");
  return clonedAddr;
}

//===----------------------------------------------------------------------===//
//                            OwnershipRAUWHelper
//===----------------------------------------------------------------------===//

OwnershipRAUWHelper::OwnershipRAUWHelper(OwnershipFixupContext &inputCtx,
                                         SILValue inputOldValue,
                                         SILValue inputNewValue)
    : ctx(&inputCtx), oldValue(inputOldValue), newValue(inputNewValue) {
  // If we are already not valid, just bail.
  if (!isValid())
    return;

  // If we are not in ownership, we can always RAUW successfully so just bail
  // and leave the object valid.
  if (!oldValue->getFunction()->hasOwnership())
    return;

  // This utility currently only handles erasing SingleValueInstructions and
  // terminator results.
  assert(isa<SingleValueInstruction>(inputOldValue)
         || cast<SILPhiArgument>(inputOldValue)->isTerminatorResult());

  // Precondition: If \p oldValue is a BorrowedValue that introduces a local
  // borrow scope, then \p newValue must either be defined in the same block as
  // \p oldValue, or it must dominate \p oldValue (rather than merely
  // dominating its uses).
  //
  // Handling cases where the new value does not dominate the old borrow scope
  // would require signficant complexity and such cases are currently impossible
  // to test. Consideration would be required for handling a new value within an
  // inner loop, while the old borrow scope is introduced outside that
  // loop. Since it generally makes no sense to do this kind of replacement,
  // we simply rule it out as an RAUW precondition.
  //
  // TODO: this could be converted to a bailout if we don't want the client code
  // to explicitly check this case. But then we may want DominanceInfo to be
  // available, which could cheaper in extreme cases because it caches results.
  SWIFT_ASSERT_ONLY_DECL(auto borrowedVal = BorrowedValue(inputOldValue));
  assert((!borrowedVal.isLocalScope()
          || checkReachingBlockDominates(inputNewValue->getParentBlock(),
                                         inputOldValue->getParentBlock()))
         && "OSSA RAUW requires reachability and dominance");

  // Clear the context before populating it anew.
  ctx->clear();

  // Otherwise, lets check if we can perform this RAUW operation. If we can't,
  // set ctx to nullptr to invalidate the helper and return.
  if (!canFixUpOwnershipForRAUW(oldValue, newValue, inputCtx)) {
    invalidate();
    return;
  }

  // If we have an object, at this point we are good to go so we can just
  // return.
  if (newValue->getType().isObject())
    return;

  // But if we have an address, we need to check if new value is from an
  // interior pointer or not in a way that the pass understands. What we do is:
  //
  // 1. Early exit some cases that we know can never have interior pointers.
  //
  // 2. Compute the AccessPathWithBase of newValue. If we do not get back a
  //    valid such object, invalidate and then bail.
  //
  // 3. Then we check if the base address is the result of an interior pointer
  //    instruction. If we do not find one we bail.
  //
  // 4. Then grab the base value of the interior pointer operand. We only
  //    support cases where we have a single BorrowedValue as our base. This is
  //    a safe future proof assumption since one reborrows are on
  //    structs/tuple/destructures, a guaranteed value will always be associated
  //    with a single BorrowedValue, so this will never fail (and the code will
  //    probably be DCEed).
  //
  // 5. Then we compute an AccessPathWithBase for oldValue and then find its
  //    derived uses. If we fail, we bail.
  //
  // 6. At this point, we know that we can perform this RAUW. The only question
  //    is if we need to when we RAUW copy the interior pointer base value. We
  //    perform this check by making sure all of the old value's derived uses
  //    are within our BorrowedValue's scope. If so, we clear the extra state we
  //    were tracking (the interior pointer/oldValue's transitive uses), so we
  //    perform just a normal RAUW (without inserting the copy) when we RAUW.
  //
  // We can always RAUW an address with a pointer_to_address since if there
  // were any interior pointer constraints on whatever address pointer came
  // from, the address_to_pointer producing that value erases that
  // information, so we can RAUW without worrying.
  //
  // NOTE: We also need to handle this here since a pointer_to_address is not a
  // valid base value for an access path since it doesn't refer to any storage.
  BorrowedAddress borrowedAddress(newValue);
  if (!borrowedAddress.mayBeBorrowed)
    return;

  if (!borrowedAddress.interiorPointerOp) {
    invalidate();
    return;
  }

  ctx->extraAddressFixupInfo.intPtrOp = borrowedAddress.interiorPointerOp;
  auto borrowedValue = borrowedAddress.interiorPointerOp.getSingleBaseValue();
  if (!borrowedValue) {
    invalidate();
    return;
  }

  // For now, just gather up uses
  auto &oldValueUses = ctx->extraAddressFixupInfo.allAddressUsesFromOldValue;
  if (InteriorPointerOperand::findTransitiveUsesForAddress(oldValue,
                                                           oldValueUses)) {
    invalidate();
    return;
  }

  // Ok, at this point we know that we can optimize. The only question is if we
  // need to perform the copy or not when we actually RAUW. So perform the is
  // within region check. If we succeed, clear our extra state so we perform a
  // normal RAUW.
  SmallVector<Operand *, 8> scratchSpace;
  if (borrowedValue.areUsesWithinScope(oldValueUses, scratchSpace,
                                       ctx->deBlocks)) {
    // We do not need to copy the base value! Clear the extra info we have.
    ctx->extraAddressFixupInfo.clear();
    return;
  }
  // This cloner check must match the later cloner invocation in
  // getReplacementAddress()
  auto *intPtrInst =
    cast<SingleValueInstruction>(ctx->extraAddressFixupInfo.intPtrOp.getUser());
  auto checkBase = [&](SILValue srcAddr) {
    return (srcAddr == intPtrInst) ? SILValue(intPtrInst) : SILValue();
  };
  if (!canCloneUseDefChain(newValue, checkBase)) {
    invalidate();
    return;
  }
}

SILValue OwnershipRAUWHelper::prepareReplacement(SILValue rewrittenNewValue) {
  assert(isValid() && "OwnershipRAUWHelper invalid?!");

  if (rewrittenNewValue) {
    // Everything about \n newValue that the constructor checks should also be
    // true for rewrittenNewValue.
    assert(rewrittenNewValue->getType() == newValue->getType());
    assert(rewrittenNewValue->getOwnershipKind()
           == newValue->getOwnershipKind());
    assert(rewrittenNewValue->getParentBlock() == newValue->getParentBlock());
    assert(BorrowedAddress(rewrittenNewValue) == BorrowedAddress(newValue));

    newValue = rewrittenNewValue;
  }
  assert(newValue && "prepareReplacement can only be called once");
  SWIFT_DEFER { newValue = SILValue(); };

  if (!oldValue->getFunction()->hasOwnership())
    return newValue;

  if (oldValue->getType().isAddress()) {
    return getReplacementAddress();
  }
  OwnershipRAUWPrepare rauwPrepare{oldValue, *ctx};
  return rauwPrepare.prepareReplacement(newValue);
}

SILBasicBlock::iterator
OwnershipRAUWHelper::perform(SILValue replacementValue) {
  if (!replacementValue)
    replacementValue = prepareReplacement();

  assert(!newValue && "prepareReplacement() must be called");

  // Make sure to always clear our context after we transform.
  SWIFT_DEFER { ctx->clear(); };

  if (auto *svi = dyn_cast<SingleValueInstruction>(oldValue))
    return replaceAllUsesAndErase(svi, replacementValue, ctx->callbacks);

  // The caller must rewrite the terminator after RAUW.
  auto *term = cast<SILPhiArgument>(oldValue)->getTerminatorForResult();
  auto nextII = term->getParent()->end();
  return replaceAllUses(oldValue, replacementValue, nextII, ctx->callbacks);
}

//===----------------------------------------------------------------------===//
//                           Single Use Replacement
//===----------------------------------------------------------------------===//

namespace {

/// Given a use and a new value, lifetime extend new value as appropriate so we
/// can replace use->get() with newValue and preserve ownership invariants. We
/// assume that old value will be left alone and not deleted so we insert
/// compensating cleanups.
struct SingleUseReplacementUtility {
  Operand *use;
  SILValue newValue;
  OwnershipFixupContext &ctx;

  SILBasicBlock::iterator handleUnowned();
  SILBasicBlock::iterator handleOwned();
  SILBasicBlock::iterator handleGuaranteed();

  SILBasicBlock::iterator perform();

  OwnershipLifetimeExtender getLifetimeExtender() { return {ctx}; }

  const InstModCallbacks &getCallbacks() const { return ctx.callbacks; }
};

} // anonymous namespace

SILBasicBlock::iterator SingleUseReplacementUtility::handleUnowned() {
  auto &callbacks = ctx.callbacks;
  switch (newValue.getOwnershipKind()) {
  case OwnershipKind::None:
    llvm_unreachable("Should have been handled elsewhere");
  case OwnershipKind::Any:
    llvm_unreachable("Invalid for values");
  case OwnershipKind::Unowned:
    // An unowned value can always be RAUWed with another unowned value.
    return replaceSingleUse(use, newValue, callbacks);
  case OwnershipKind::Guaranteed: {
    // If we have an unowned value use that we want to replace with a guaranteed
    // value, we need to ensure that the guaranteed value is live at that use
    // point. If we know that is always true, just perform the replace.
    //
    // FIXME: Expand the cases here.
    if (isa<SILFunctionArgument>(newValue))
      return replaceSingleUse(use, newValue, callbacks);

    // Otherwise, we need to lifetime extend newValue to the use. If the actual
    // use is a terminator, we need to insert an unchecked_ownership_conversion
    // since our value can not be live at the terminator itself.
    if (auto *ti = dyn_cast<TermInst>(use->getUser())) {
      if (ti->isFunctionExiting()) {
        SILBuilderWithScope builder(ti);
        auto *newInst = builder.createUncheckedOwnershipConversion(
            ti->getLoc(), use->get(), OwnershipKind::Unowned);
        callbacks.createdNewInst(newInst);
        callbacks.setUseValue(use, newInst);
      }
    }

    auto extender = getLifetimeExtender();
    SILValue borrow = extender.borrowOverSingleUse(newValue, use);
    assert(!use->isLifetimeEnding()
           && "Test single-use replacement of a scope-ending instruction");

    return replaceSingleUse(use, borrow, callbacks);
  }
  case OwnershipKind::Owned: {
    // If we have an unowned value use that we want to replace with an owned
    // value use.  we first check if the owned value is live over all use points
    // of the old value. If so, just RAUW and continue.
    //
    // TODO: Implement this.

    // Otherwise, insert a copy of the owned value and lifetime extend that over
    // the use.
    //
    // NOTE: For terminator uses, we funnel the use through an
    // unchecked_ownership_conversion to ensure that we can end the lifetime of
    // our owned/guaranteed value before the terminator.
    if (auto *ti = dyn_cast<TermInst>(use->getUser())) {
      if (ti->isFunctionExiting()) {
        SILBuilderWithScope builder(ti);
        auto *newInst = builder.createUncheckedOwnershipConversion(
            ti->getLoc(), use->get(), OwnershipKind::Unowned);
        callbacks.createdNewInst(newInst);
        callbacks.setUseValue(use, newInst);
      }
    }

    auto extender = getLifetimeExtender();
    SILValue copy = extender.createPlusZeroCopy(newValue, {use});
    return replaceSingleUse(use, copy, callbacks);
  }
  }
  llvm_unreachable("covered switch isn't covered?!");
}

SILBasicBlock::iterator SingleUseReplacementUtility::handleGuaranteed() {
  // Ok, our use is guaranteed and our new value may not be guaranteed.
  auto extender = getLifetimeExtender();

  // If our original use was a lifetime ending use...
  //
  // TODO: Also just call borrowOverSingleUse in the lifetime-ending case.
  if (use->isLifetimeEnding()) {
    // And additionally was a reborrow, we will have placed it in recursive
    // reborrows. In this case the RAUW
    if (ctx.recursiveReborrows.size()) {
      eliminateReborrowsOfRecursiveBorrows(
          ctx.recursiveReborrows, ctx.transitiveBorrowedUses, ctx.callbacks);
      // By eliminate the reborrows our lifetime ending use was already
      // handled. Now, we need to lifetime extend the borrow over all of the
      // end_lifetime after we eliminate the reborrow. These will be the
      // transitive borrowed uses.
      auto borrowPt = newValue->getNextInstruction()->getIterator();
      SILValue borrow = extender.borrowCopyOverGuaranteedUses(
          newValue, borrowPt, llvm::makeArrayRef(ctx.transitiveBorrowedUses));

      // Then replace use->get() with this borrow.
      return replaceSingleUse(use, borrow, ctx.callbacks);
    } else {
      // If we didn't have a reborrow and still had a lifetime ending use,
      // handle it.
      SILValue borrow =
          extender.originalCreatePlusOneBorrow(newValue, use->getUser());
      // Then replace use->get() with this copy. We will insert compensating end
      // scope instructions on use->get() if we need to.
      return replaceSingleUse(use, borrow, ctx.callbacks);
    }
  }

  // If we don't have a lifetime ending use, just create the borrow.
  SILValue copy = extender.borrowOverSingleUse(newValue, use);

  // Then replace use->get() with this copy. We will insert compensating end
  // scope instructions on use->get() if we need to.
  return replaceSingleUse(use, copy, ctx.callbacks);
}

SILBasicBlock::iterator SingleUseReplacementUtility::handleOwned() {
  // Ok, our old value is owned and our new value may not be owned. First
  // lifetime extend newValue to use->getUser() inserting destroy_values along
  // any paths that do not go through use->getUser().
  auto extender = getLifetimeExtender();

  if (use->isLifetimeEnding()) {
    // If our use is a lifetime ending use, then create a plus one copy and
    // RAUW.
    SILValue copy = extender.createPlusOneCopy(newValue, use->getUser());
    // Then replace use->get() with this copy. We will insert compensating end
    // scope instructions on use->get() if we need to.
    return replaceSingleUse(use, copy, ctx.callbacks);
  }

  // If we don't have a lifetime ending use, just create a +0 copy and set the
  // use. All destroys will be placed for us.
  SILValue copy =
      extender.createPlusZeroCopy<ArrayRef<Operand *>>(newValue, {use});

  // Then replace use->get() with this copy. We will insert compensating end
  // scope instructions on use->get() if we need to.
  return replaceSingleUse(use, copy, ctx.callbacks);
}

SILBasicBlock::iterator SingleUseReplacementUtility::perform() {
  auto oldValue = use->get();
  assert(oldValue->getFunction()->hasOwnership());

  // If our new value is just none, we can pass anything to do it so just RAUW
  // and return.
  //
  // NOTE: This handles RAUWing with undef.
  if (newValue.getOwnershipKind() == OwnershipKind::None)
    return replaceSingleUse(use, newValue, ctx.callbacks);

  assert(SILValue(oldValue).getOwnershipKind() != OwnershipKind::None);

  switch (SILValue(oldValue).getOwnershipKind()) {
  case OwnershipKind::None:
    // If our old value was none and our new value is not, we need to do
    // something more complex that we do not support yet, so bail. We should
    // have not called this function in such a case.
    llvm_unreachable("Should have been handled elsewhere");
  case OwnershipKind::Any:
    llvm_unreachable("Invalid for values");
  case OwnershipKind::Guaranteed:
    return handleGuaranteed();
  case OwnershipKind::Owned:
    return handleOwned();
  case OwnershipKind::Unowned:
    return handleUnowned();
  }
  llvm_unreachable("Covered switch isn't covered?!");
}

//===----------------------------------------------------------------------===//
//                      OwnershipReplaceSingleUseHelper
//===----------------------------------------------------------------------===//

OwnershipReplaceSingleUseHelper::OwnershipReplaceSingleUseHelper(
    OwnershipFixupContext &inputCtx, Operand *inputUse, SILValue inputNewValue)
    : ctx(&inputCtx), use(inputUse), newValue(inputNewValue) {
  // If we are already not valid, just bail.
  if (!isValid())
    return;

  // If we do not have ownership, we are already done.
  if (!inputUse->getUser()->getFunction()->hasOwnership())
    return;

  // If we have an address, bail. We don't support this.
  if (newValue->getType().isAddress()) {
    invalidate();
    return;
  }

  // Otherwise, lets check if we can perform this RAUW operation. If we can't,
  // set ctx to nullptr to invalidate the helper and return.
  if (!OwnershipRAUWHelper::hasValidRAUWOwnership(use->get(), newValue)) {
    invalidate();
    return;
  }

  // FIXME:!!! If this does not use canFixUpOwnershipForRAUW, then it needs to
  // do the equivalent safety checks. At least ensure that the use is not a
  // PointerEscape. But we should put that check behind a standard utility.

  // Then see if our use is a lifetime ending use of a guaranteed value that is
  // a reborrow.
  if (auto reborrowOperand = BorrowingOperand(use)) {
    if (reborrowOperand.isReborrow()) {
      // Check that the old lifetime can be extended and record the necessary
      // book-keeping in the OwnershipFixupContext.
      ctx->recursiveReborrows.push_back(use);
    }
  }
}

SILBasicBlock::iterator OwnershipReplaceSingleUseHelper::perform() {
  assert(isValid() && "OwnershipReplaceSingleUseHelper invalid?!");

  if (!use->getUser()->getFunction()->hasOwnership())
    return replaceSingleUse(use, newValue, ctx->callbacks);

  // Make sure to always clear our context after we transform.
  SWIFT_DEFER { ctx->clear(); };
  SingleUseReplacementUtility utility{use, newValue, *ctx};
  return utility.perform();
}

//===----------------------------------------------------------------------===//
//                      createBorrowScopeForPhiOperands
//===----------------------------------------------------------------------===//

/// Given a phi that has been newly created or converted from terminator
/// results, check for inner guaranteed operands (which do not introduce a
/// borrow scope). This is invalid OSSA because the phi is a reborrow, and all
/// borrow-scope-ending instructions must directly use the BorrowedValue that
/// introduces the scope.
///
/// Create nested borrow scopes for its operands.
///
/// Transitively follow its phi uses.
///
/// Create end_borrows at all points that cover the inner uses.
///
/// The client must check canCloneTerminator() first to make sure that the
/// search for transitive uses does not encouter a PointerEscape.
class GuaranteedPhiBorrowFixup {
  // A phi in mustConvertPhis has already been determined to be part of this
  // new nested borrow scope.
  SmallSetVector<SILPhiArgument *, 8> mustConvertPhis;

  // Phi operands that are already within the new nested borrow scope.
  llvm::SmallDenseSet<PhiOperand, 8> nestedPhiOperands;

public:
  /// Return true if an extended nested borrow scope was created.
  bool createExtendedNestedBorrowScope(SILPhiArgument *newPhi);

protected:
  bool phiOperandNeedsBorrow(Operand *operand) {
    SILValue inVal = operand->get();
    if (inVal.getOwnershipKind() != OwnershipKind::Guaranteed) {
      assert(inVal.getOwnershipKind() == OwnershipKind::None);
      return false;
    }
    // This operand needs a nested borrow if inVal is not a BorrowedValue.
    return !bool(BorrowedValue(inVal));
  }

  void borrowPhiOperand(Operand *oper) {
    // Begin the borrow just before the branch.
    SILInstruction *borrowPoint = oper->getUser();
    auto loc = RegularLocation::getAutoGeneratedLocation(borrowPoint->getLoc());
    auto *borrow =
        SILBuilderWithScope(borrowPoint).createBeginBorrow(loc, oper->get());
    oper->set(borrow);
  }

  EndBorrowInst *createEndBorrow(SILValue guaranteedValue,
                                 SILBasicBlock::iterator borrowPoint) {
    auto loc = borrowPoint->getLoc();
    return SILBuilderWithScope(borrowPoint)
        .createEndBorrow(loc, guaranteedValue);
  }

  void insertEndBorrowsAndFindPhis(SILPhiArgument *phi);
};

void GuaranteedPhiBorrowFixup::insertEndBorrowsAndFindPhis(
    SILPhiArgument *phi) {
  // Scope ending instructions are only needed for nontrivial results.
  if (phi->getOwnershipKind() != OwnershipKind::Guaranteed) {
    assert(phi->getOwnershipKind() == OwnershipKind::None);
    return;
  }
  SmallVector<Operand *, 16> usePoints;
  bool result = findInnerTransitiveGuaranteedUses(phi, usePoints);
  assert(result && "should be checked by canCloneTerminator");
  (void)result;

  // Add usePoints to a set for phi membership checking.
  //
  // FIXME: consider integrating with ValueLifetimeBoundary instead.
  SmallPtrSet<Operand *, 16> useSet(usePoints.begin(), usePoints.end());

  auto phiUsers = llvm::map_range(usePoints, ValueBase::UseToUser());
  ValueLifetimeAnalysis lifetimeAnalysis(phi, phiUsers);
  ValueLifetimeBoundary boundary;
  lifetimeAnalysis.computeLifetimeBoundary(boundary);

  for (auto *boundaryEdge : boundary.boundaryEdges) {
    createEndBorrow(phi, boundaryEdge->begin());
  }

  for (SILInstruction *lastUser : boundary.lastUsers) {
    // If the last use is a branch, transitively process the phi.
    if (isa<BranchInst>(lastUser)) {
      for (Operand &oper : lastUser->getAllOperands()) {
        if (!useSet.count(&oper))
          continue;

        PhiOperand phiOper(&oper);
        nestedPhiOperands.insert(phiOper);
        mustConvertPhis.insert(phiOper.getValue());
        continue;
      }
    }
    // If the last user is a terminator, add the successors as boundary edges.
    if (isa<TermInst>(lastUser)) {
      for (auto *succBB : lastUser->getParent()->getSuccessorBlocks()) {
        // succBB cannot already be in boundaryEdges. It has a
        // single predecessor with liveness ending at the terminator, which
        // means it was not live into any successor blocks.
        createEndBorrow(phi, succBB->begin());
      }
      continue;
    }
    // Otherwise, just plop down an end_borrow after the last use.
    createEndBorrow(phi, std::next(lastUser->getIterator()));
  }
};

// For each phi that transitively uses an inner guaranteed value, create nested
// borrow scopes so that it is a well-formed reborrow.
bool GuaranteedPhiBorrowFixup::
createExtendedNestedBorrowScope(SILPhiArgument *newPhi) {
  // Determine if this new phi needs a nested borrow scope. If so, seed the
  // Visit phi operands, returning false as soon as one needs a borrow.
  if (!newPhi->visitIncomingPhiOperands(
        [&](Operand *op) { return !phiOperandNeedsBorrow(op); })) {
    mustConvertPhis.insert(newPhi);
  }
  if (mustConvertPhis.empty())
    return false;

  // mustConvertPhis grows in this loop.
  for (unsigned mustConvertIdx = 0; mustConvertIdx < mustConvertPhis.size();
         ++mustConvertIdx) {
    SILPhiArgument *phi = mustConvertPhis[mustConvertIdx];
    insertEndBorrowsAndFindPhis(phi);
  }
  // To handle recursive phis, first discover all phis before attempting to
  // borrow any phi operands.
  for (SILPhiArgument *phi : mustConvertPhis) {
    phi->visitIncomingPhiOperands([&](Operand *op) {
      if (!nestedPhiOperands.count(op))
        borrowPhiOperand(op);
      return true;
    });
  }
  return true;
}

// Note: \p newPhi itself might not have Guaranteed ownership. A phi that
// converts Guaranteed to None ownership still needs nested borrows.
//
// Note: This may be called on partially invalid OSSA form, where multiple
// newly created phis do not yet have a borrow scope. The implementation
// assumes that this API will eventually be called for all such new phis until
// OSSA is fully valid.
bool swift::createBorrowScopeForPhiOperands(SILPhiArgument *newPhi) {
  return GuaranteedPhiBorrowFixup().createExtendedNestedBorrowScope(newPhi);
}
