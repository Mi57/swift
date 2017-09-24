//===--- BlockArgumentStorage.h - Block Argument Storage Optimizer --------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2018 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

/// An analysis used by AddressLowering to reuse storage across block arguments.
///
/// Populates Result::projectedBBArgs with all inputs to bbArg that can reuse
/// the argument's storage.
class BlockArgumentStorageOptimizer {
  class Result {
    friend class BlockArgumentStorageOptimizer;
    SmallVector<Operand *, 4> projectedBBArgs;

    struct GetOper {
      SILValue operator()(Operand *oper) const { return oper->get(); }
    };

    Result(const Result &) = delete;
    Result& operator=(const Result&) = delete;

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

  AddressLoweringState &pass;

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
  BlockArgumentStorageOptimizer(AddressLoweringState &pass,
                                SILPHIArgument *bbArg)
      : pass(pass), bbArg(bbArg) {}

  Result &&computeArgumentProjections() &&;

protected:
  bool computeIncomingLiveness(Operand *useOper, SILBasicBlock *defBB);
  bool hasCoalescedOperand(SILInstruction *defInst);
};
