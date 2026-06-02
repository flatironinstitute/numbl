% TEST: 1 .^ Inf at the top level.
% opt0 (interp, before fix): RuntimeError "Maximum call stack size exceeded"
%   <-- DIVERGED (infinite recursion in binop)
% opt1 (JS-JIT): NaN
% opt2 (C-JIT): NaN (mtoc2_pow_real matches Math.pow)
% DIVERGING MODE: opt0 only (crash).
%
% Cause: binop's numeric fast path computed Math.pow(1, Inf) = NaN and broke
%   to the slow path, but the secondary fast path re-entered binop with the
%   same two plain numbers, recursing forever. FIX (runtimeOperators.ts):
%   skip the secondary fast path when both operands are already plain
%   numbers. All three modes now agree on NaN.
disp(1 .^ Inf);
