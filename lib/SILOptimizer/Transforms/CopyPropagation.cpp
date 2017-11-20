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
/// Copy propagation removed unnecessary copy_value/destroy_value instructions.
/// [WIP] This is meant to complement opaque values. Initially at -O, but
/// eventually may also be adapted to -Onone (as currently designed, it shrinks
/// variable live ranges).
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
///    - Recurse through aggregates:
///      (Struct, Tuple, Enum, InitExistential).
///    - Track the current projection path ID.
///    - Recurse through some extractions, stripping the path ID.
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
///    (minor destroy materialization details omitted here)
/// 
/// ===----------------------------------------------------------------------===
