% TEST: a function (and a loop inside one) first run with a complex tensor
% input, then with a REAL tensor that reuses the complex specialization.
% The tensor counterpart of A38.
% opt0 (interp): values below
% opt1 (JS-JIT): "Internal JIT error in jit-call: Cannot read properties of
%   undefined (reading '0')" and, for the loop, "Internal JIT error in
%   jit-loop: Cannot set properties of undefined (setting '2')"
% opt2 (C-JIT):  values below
% DIVERGING MODE: opt1 only. MATLAB gold = opt0.
%
% Cause: type widening makes the real tensor input complex-typed, but the
%   JS value adapter numblToJit (executors/jit/valueAdapter.ts) passed a
%   real tensor without an `imag` lane, so the complex-typed body read or
%   wrote `.imag` of undefined. The C adapter gives it a zero imag lane.
%   Before the raise channel (#118) the TypeError was swallowed as a bail.
% JIT engagement: CONFIRMED.
function y = dbl(x)
  y = x;
  for k = 1:numel(x)
    y(k) = 2*x(k);
  end
end
function p = recur(x)
  p = zeros(numel(x), 3);
  p(:, 1) = x;
  for k = 2:3
    p(:, k) = k * p(:, k-1);
  end
end
try;catch;end
disp(dbl([1+2i; 3]))
y = dbl([1; 3]);
disp(y)
disp(isequal(y, [2; 6]))
disp(recur([1i; 2]))
p = recur([1; 2]);
disp(p)
disp(isequal(p, [1 2 6; 2 4 12]))
