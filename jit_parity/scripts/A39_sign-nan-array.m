% TEST: sign() of an ARRAY containing NaN.
% MATLAB: sign(NaN) is NaN, so sign([NaN 1]) = [NaN 1].
% opt0 (interp): 0   1     <-- DIVERGES (NaN -> 0)
% opt1 (JS-JIT): NaN   1
% opt2 (C-JIT):  NaN   1
% DIVERGING MODE: opt0 only (the interpreter is the wrong one here).
%
% Cause: the array path of sign() in the interpreter dispatches to the
%   native unary op (opcode 19), whose custom helper rsign in
%   ops/realUnaryElemwise.ts is `x>0?1:x<0?-1:0` -> NaN falls through both
%   comparisons and returns 0. The native C twin native/ops/
%   real_unary_elemwise.c has the identical bug. (Scalar sign(NaN) goes
%   through Math.sign and is correct; only the tensor kernel is wrong.)
%   The JIT snippets compute sign correctly, so opt1/opt2 match MATLAB.
% (x is a function param so the value isn't compile-time folded.)
function r=f(x)
  r=sign(x);
end
disp(f([NaN 1]))
