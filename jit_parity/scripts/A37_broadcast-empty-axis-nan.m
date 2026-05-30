% TEST: implicit-expansion elementwise op where one operand has a 0-size
% axis and the other a 1-size axis: zeros(0,1) + [10 20 30].
% MATLAB: the result is empty 0x3 (a 0-size axis stays 0).
% opt0 (interp): empty (before / after, nothing between)
% opt1 (JS-JIT): "   NaN   NaN   NaN" (a spurious 1x3 of NaN)  <-- DIVERGES
% opt2 (C-JIT):  empty (correct)
% DIVERGING MODE: opt1 only (opt0==opt2).
%
% Cause: tensor_ops/tensor_elemwise_real.js computes the broadcast output
%   shape as outShape[i] = Math.max(ashape[i], bshape[i]); Math.max(0,1)=1
%   is wrong -- a 0 axis must win. The C twin tensor_elemwise_real.h does
%   (adim==1)?bdim:adim correctly, so opt2 is right. Same bug in the
%   _complex and _real_fn JS variants. Silent wrong shape + NaN data.
% JIT engagement: CONFIRMED (C-JIT engaged and correct; only JS diverges).
x = zeros(0,1) + [10 20 30];
disp('before'); disp(x); disp('after');
