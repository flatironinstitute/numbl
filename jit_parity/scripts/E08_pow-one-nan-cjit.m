% TEST: 1 .^ NaN with the scalar real power compiled by C-JIT.
% MATLAB / opt0 (interp, Math.pow): NaN.
% opt1 (JS-JIT, Math.pow): NaN.
% opt2 (C-JIT, before fix): 1   <-- DIVERGED (bare C99 pow(1, NaN) == 1)
% DIVERGING MODE: opt2 only (silent wrong value).
%
% Cause: power.ts emitC emitted bare `pow(a,b)`, but C99 pow returns 1 for
%   |base|==1 with a non-finite exponent, whereas ECMAScript Math.pow (used
%   by the interpreter and JS-JIT) returns NaN. FIX: emit a `mtoc2_pow_real`
%   helper that matches Math.pow, and use it in the scalar path and the
%   tensor power kernels.
%
% NOTE: 1 .^ Inf would stack-overflow the interpreter (a separate opt0 bug),
%   so this script uses only the NaN exponent.
%!numbl:assert_jit c
b = 1 .^ NaN;
disp(b);
