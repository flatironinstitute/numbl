% TEST: `NaN && x` and `NaN || x` where NaN is a compile-time-known LHS.
% numbl treats a nonzero NaN as TRUTHY everywhere (`!= 0` in C, `!== 0` in
% JS, `toBool` in the interpreter — see `truthy()` in emitJs and the
% andand/oror builtins; `while NaN` runs in all modes).
% opt0 (interp): prints "1 0 1"   (NaN truthy: NaN&&1 -> 1, NaN&&0 -> 0,
%                                   NaN||0 -> 1)
% opt1 (JS-JIT): prints "0 0 1"   (WRONG: NaN&&x folded to false)
% opt2 (C-JIT):  prints "0 0 1"   (WRONG: same fold)
% DIVERGES: opt1 + opt2 (first two columns) vs opt0.
% CAUSE: the short-circuit fold in lowering/lower.ts folded `NaN && x` to
%   false (`lhsExact === 0 || Number.isNaN(lhsExact)`), treating NaN as
%   falsy — even though the `||` arm already EXCLUDED NaN and the andand
%   builtin computes NaN as truthy. Fixed by dropping the NaN arm from the
%   `&&` fold so `NaN && x` defers to the builtin's RHS evaluation,
%   matching `NaN || x` and the interpreter.
% JIT-ENGAGEMENT: all-fprintf void script, no unsuppressed output -> whole
%   scope JIT-compiles; the fold fires because `n = NaN` is a known literal.
n = NaN;
fprintf('%d %d %d\n', n && 1, n && 0, n || 0);
