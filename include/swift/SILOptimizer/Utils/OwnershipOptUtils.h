//===--- OwnershipOptUtils.h ----------------------------------------------===//
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

#ifndef SWIFT_SILOPTIMIZER_UTILS_OWNERSHIPOPTUTILS_H
#define SWIFT_SILOPTIMIZER_UTILS_OWNERSHIPOPTUTILS_H

#include "swift/Basic/Defer.h"
#include "swift/SIL/BasicBlockUtils.h"
#include "swift/SIL/OwnershipUtils.h"
#include "swift/SIL/SILModule.h"
#include "swift/SILOptimizer/Utils/InstOptUtils.h"

namespace swift {

// Defined in BasicBlockUtils.h
struct JointPostDominanceSetComputer;

/// Model an extended borrow scope, including transitive reborrows. Allow
/// extending the lifetime of an owned value across that extended borrow
/// scope. This handles uses of reborrows that are not dominated by the owned
/// value by generating phis and copying the borrowed values the reach this
/// borrow scope from non-dominated paths.
///
/// This applies to "local" borrow scopes (begin_borrow, load_borrow, & phi).
struct BorrowedLifetimeExtender {
  /// If the extended lifetime of \p borrowedValue can be analyzed, return a
  /// valid BorrowedLifetimeExtender instance with the information needed to
  /// extend an owned value over that extended lifetime.
  ///
  /// A borrowed value's "extended" lifetime transitively include the uses of
  /// reborrows.
  static BorrowedLifetimeExtender
  canExtendOverBorrowScope(BorrowedValue borrowedValue);

private:
  BorrowedValue borrowedValue;

  llvm::SmallDenseSet<PhiOperand, 4> reborrowedOperands;
  llvm::SmallSetVector<PhiValue, 4> reborrowedValues;

  BorrowedLifetimeExtender() = default;

  BorrowedLifetimeExtender(BorrowedValue borrowedValue)
      : borrowedValue(borrowedValue) {}

  bool computeExtensionOverBorrowScope();

public:
  operator bool() const { return borrowedValue; }

  /// Extend \p ownedValue over this extended borrow scope
  void extendOverBorrowScope(SILValue ownedValue);
};

/// A struct that contains context shared in between different operation +
/// "ownership fixup" utilities. Please do not put actual methods on this, it is
/// meant to be composed with.
struct OwnershipFixupContext {
  Optional<InstModCallbacks> inlineCallbacks;
  InstModCallbacks &callbacks;
  DeadEndBlocks &deBlocks;
  JointPostDominanceSetComputer &jointPostDomSetComputer;

  SmallVector<Operand *, 8> transitiveBorrowedUses;
  SmallVector<PhiOperand, 8> recursiveReborrows;

  /// Extra state initialized by OwnershipRAUWFixupHelper::get() that we use
  /// when RAUWing addresses. This ensures we do not need to recompute this
  /// state when we perform the actual RAUW.
  struct AddressFixupContext {
    /// When determining if we need to perform an address pointer fixup, we
    /// compute all transitive address uses of oldValue. If we find that we do
    /// need this fixed up, then we will copy our interior pointer base value
    /// and use this to seed that new lifetime.
    SmallVector<Operand *, 8> allAddressUsesFromOldValue;

    /// This is the interior pointer operand that the new value we want to RAUW
    /// is transitively derived from and enables us to know the underlying
    /// borrowed base value that we need to lifetime extend.
    InteriorPointerOperand intPtrOp;

    void clear() {
      allAddressUsesFromOldValue.clear();
      intPtrOp = InteriorPointerOperand();
    }
  };
  AddressFixupContext extraAddressFixupInfo;

  OwnershipFixupContext(InstModCallbacks &callbacks, DeadEndBlocks &deBlocks,
                        JointPostDominanceSetComputer &jointPostDomSetComputer)
      : callbacks(callbacks), deBlocks(deBlocks),
        jointPostDomSetComputer(jointPostDomSetComputer) {}

  void clear() {
    jointPostDomSetComputer.clear();
    transitiveBorrowedUses.clear();
    recursiveReborrows.clear();
    extraAddressFixupInfo.allAddressUsesFromOldValue.clear();
    extraAddressFixupInfo.intPtrOp = InteriorPointerOperand();
  }

  /// Gets access to the joint post dominance computer and clears it after \p
  /// callback.
  template <typename ResultTy>
  ResultTy withJointPostDomComputer(
      function_ref<ResultTy(JointPostDominanceSetComputer &)> callback) {
    // Make sure we clear the joint post dom computer after callback.
    SWIFT_DEFER { jointPostDomSetComputer.clear(); };
    // Then return callback passing in the computer.
    return callback(jointPostDomSetComputer);
  }

private:
  /// Helper method called to determine if we discovered we needed interior
  /// pointer fixups while simplifying.
  bool needsInteriorPointerFixups() const {
    return bool(extraAddressFixupInfo.intPtrOp);
  }
};

/// A utility composed ontop of OwnershipFixupContext that knows how to RAUW a
/// value or a single value instruction with a new value and then fixup
/// ownership invariants afterwards.
class OwnershipRAUWHelper {
  OwnershipFixupContext *ctx;
  SingleValueInstruction *oldValue;
  SILValue newValue;

public:
  OwnershipRAUWHelper() : ctx(nullptr), oldValue(nullptr), newValue(nullptr) {}

  /// Return an instance of this class if we can perform the specific RAUW
  /// operation ignoring if the types line up. Returns None otherwise.
  ///
  /// DISCUSSION: We do not check that the types line up here so that we can
  /// allow for our users to transform our new value in ways that preserve
  /// ownership at \p oldValue before we perform the actual RAUW. If \p newValue
  /// is an object, any instructions in the chain of transforming instructions
  /// from \p newValue at \p oldValue's must be forwarding. If \p newValue is an
  /// address, then these transforms can only transform the address into a
  /// derived address.
  OwnershipRAUWHelper(OwnershipFixupContext &ctx,
                      SingleValueInstruction *oldValue, SILValue newValue);

  /// Returns true if this helper was initialized into a valid state.
  operator bool() const { return isValid(); }
  bool isValid() const { return bool(ctx) && bool(oldValue) && bool(newValue); }

  /// Perform the actual RAUW. We require that \p newValue and \p oldValue have
  /// the same type at this point (in contrast to when calling
  /// OwnershipRAUWFixupHelper::get()).
  ///
  /// This is so that we can avoid creating "forwarding" transformation
  /// instructions before we know if we can perform the RAUW. Any such
  /// "forwarding" transformation must be performed upon \p newValue at \p
  /// oldValue's insertion point so that we can then here RAUW the transformed
  /// \p newValue.
  SILBasicBlock::iterator
  perform(SingleValueInstruction *maybeTransformedNewValue = nullptr);

private:
  SILBasicBlock::iterator replaceAddressUses(SingleValueInstruction *oldValue,
                                             SILValue newValue);
};

/// A utility composed ontop of OwnershipFixupContext that knows how to replace
/// a single use of a value with another value with a different ownership. We
/// allow for the values to have different types.
///
/// NOTE: When not in OSSA, this just performs a normal set use, so this code is
/// safe to use with all code.
class OwnershipReplaceSingleUseHelper {
  OwnershipFixupContext *ctx;
  Operand *use;
  SILValue newValue;

public:
  OwnershipReplaceSingleUseHelper()
      : ctx(nullptr), use(nullptr), newValue(nullptr) {}

  /// Return an instance of this class if we support replacing \p use->get()
  /// with \p newValue.
  ///
  /// NOTE: For now we only support objects, not addresses so addresses will
  /// always yield an invalid helper.
  OwnershipReplaceSingleUseHelper(OwnershipFixupContext &ctx, Operand *use,
                                  SILValue newValue);

  /// Returns true if this helper was initialized into a valid state.
  operator bool() const { return isValid(); }
  bool isValid() const { return bool(ctx) && bool(use) && bool(newValue); }

  /// Perform the actual RAUW.
  SILBasicBlock::iterator perform();
};

} // namespace swift

#endif
