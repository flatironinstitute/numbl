% JIT complex .* and ./ follow MATLAB's mixed real/complex rule (as the
% interpreter, see arithmetic/mixed_real_complex_nan.m): a complex operand
% with an all-zero imaginary part acts as real, and a real operand is not
% promoted to complex, so NaN or Inf in it does not make NaN imaginary parts.
% `+ 0i` gives the JIT a complex-typed operand with all-zero imaginary part
% (complex() is not JIT-compiled; MATLAB makes these operands real).

%!numbl:assert_jit c
x = [NaN; 1; Inf] + 0i; r = [2; 3; 4]; M = [1 2; 3 4; 5 6];
for k = 1:3
  [a, d] = tms_rdv(x, r);
end
assert(isequal(imag(a), [0; 0; 0]) && isreal(a));
assert(isequal(imag(d), [0; 0; 0]) && isreal(d));
assert(isequal(imag(tms(r, x)), [0; 0; 0]));
assert(isequal(imag(tms(x, M)), zeros(3, 2)));
assert(isequal(imag(tms(M, x)), zeros(3, 2)));
assert(isequal(imag(tms(x, 2)), [0; 0; 0]) && isequal(imag(tms(2, x)), [0; 0; 0]));
assert(isequal(imag(mt(x, 2)), [0; 0; 0]) && isequal(imag(mt(2, x)), [0; 0; 0]));
q = real(rdv(r, x));
assert(isnan(q(1)) && q(2) == 3 && q(3) == 0);
assert(isequal(imag(rdv(r, x)), [0; 0; 0]));
y = NaN + 0i;
assert(isreal(tms(y, r)) && isreal(tms(r, y)) && isreal(rdv(r, y)));
assert(isreal(tms([1; 2] + 0i, [1 NaN])));
assert(isreal(tms(x, [2; 3; 4] + 0i)));

% genuinely complex: imaginary part stays 0 where the operand's is 0
z = [1+1i, 2];
assert(isequal(imag(tms(z, [3 NaN])), [3 0]) && isequal(imag(tms([3 NaN], z)), [3 0]));
assert(isequal(imag(tms(z, [3 Inf])), [3 0]));
w = tms(z, NaN);
assert(isnan(imag(w(1))) && imag(w(2)) == 0);
assert(isequal(imag(rdv(z, [3 0])), [1/3 0]) && isequal(real(rdv(z, [3 0])), [1/3 Inf]));
assert(isequal(imag(rdv(z, 0)), [Inf 0]));
assert(isequal(imag(rdv([2 NaN], z)), [-1 0]));
assert(isequal(imag(rdv([1+2i, NaN], [0 2])), [Inf 0]));
w = tms([1+1i; 2], [3 NaN]);
assert(isequal(imag(w(:, 1)), [3; 0]) && isnan(imag(w(1, 2))) && imag(w(2, 2)) == 0);
w = tms(z, [3, 4+1i]);
assert(isequal(real(w), [3 8]) && isequal(imag(w), [3 2]));

disp('SUCCESS')

function [a, d] = tms_rdv(x, r)
  %!numbl:assert_jit c
  a = x .* r;
  d = x ./ r;
end

function y = tms(a, b)
  %!numbl:assert_jit c
  y = a .* b;
end

function y = rdv(a, b)
  %!numbl:assert_jit c
  y = a ./ b;
end

function y = mt(a, b)
  %!numbl:assert_jit c
  y = a * b;
end
