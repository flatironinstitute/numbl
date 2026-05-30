% TEST: subset-column assignment A(:,[2 4]) = RHS where the column index is
% an inline literal index-vector and another slot is a colon.
% opt0 (interp): correct (1 91 7 92 / 2 93 8 94 / 3 95 9 96)
% opt1 (JS-JIT): correct (matches opt0)
% opt2 (C-JIT):  RuntimeError "emit internal: IndexVec slot expr must be a
%   Var after ANF"  <-- DIVERGES (hard crash during C emit)
% DIVERGING MODE: opt2 only.
%
% Cause: lowerIndexSliceStore (lowering/lowerIndexSliceStore.ts) only ANFs a
%   LogicalMask slot to a Var; an inline-literal IndexVec slot reaches emit
%   unwrapped, and emitSliceSlotSetup (codegen/emitIndex.ts) asserts the slot
%   expr is a Var. A Colon + literal-IndexVec mix (A(:,[2 4]), A([1 3],:))
%   triggers it; a Var index (A(:,v)) or two literal IndexVec slots route
%   differently and compile fine. Common assignment pattern -> high impact.
% JIT engagement: CONFIRMED (crashes mid-emit; dump-c can't complete).
A = reshape(1:12,3,4);
A(:,[2 4]) = [91 92; 93 94; 95 96];
disp(A);
