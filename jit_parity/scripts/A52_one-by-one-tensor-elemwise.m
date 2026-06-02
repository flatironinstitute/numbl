% TEST: a genuine 1x1 tensor (ones(1,1).*5) used in an element-wise op
% against a multi-element tensor, where the loop is compiled by the
% jit-loop executor (the try/catch forces the whole-program executor to
% decline, so the 1x1 is captured as a marshaled loop INPUT).
% MATLAB / opt0 (interp): broadcasts -> 6 7 8.
% opt1 (JS-JIT, before fix): NaN NaN NaN   <-- DIVERGED
% opt2 (C-JIT, before fix):  NaN NaN NaN   <-- DIVERGED
% DIVERGING MODE: opt1 and opt2 (silent NaN).
%
% Cause: the compiler collapses a [1,1] tensor type to a scalar
%   double/complex (!isMultiElement), so fused codegen emits `a.data[i]+b`
%   for a scalar `b`, but the value adapters marshaled the 1x1 RuntimeTensor
%   as a tensor OBJECT -> number+object = NaN. FIX: unwrap a 1x1 tensor to
%   its element when the compiled param is scalar (numblToJit +
%   marshalOneInput).
b = ones(1, 1) .* 5;
a = [1 2 3];
try
    q = 1;
catch
    q = 2;
end
for k = 1:1000
    z = a + b;
end
disp(z);
