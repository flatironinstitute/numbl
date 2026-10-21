% JIT scalar complex *, /, .* and ./ follow MATLAB's mixed real/complex rule
% (as the interpreter and the JIT tensor kernels, see
% jit_mixed_real_complex_nan.m): a complex scalar with a zero imaginary part
% acts as real, and a real scalar scales each part of a complex one without
% promotion, so NaN or Inf in it does not make NaN parts. `+ 0i` gives the
% JIT a complex-typed operand with zero imaginary part (complex() is not
% JIT-compiled; MATLAB makes these operands real).

%!numbl:assert_jit c
y = NaN + 0i;
a = y*3; b = y/3; c = 3/y; d = (Inf + 0i)*2; e = 1 + Inf*1i;
assert(isnan(real(a)) && imag(a) == 0);
assert(isnan(real(b)) && imag(b) == 0);
assert(isnan(real(c)) && imag(c) == 0);
assert(real(d) == Inf && imag(d) == 0);
assert(real(e) == 1 && imag(e) == Inf);
a = y.*3; b = 3./y; c = 2*y; d = Inf/(2 + 0i);
assert(isnan(real(a)) && imag(a) == 0);
assert(isnan(real(b)) && imag(b) == 0);
assert(isnan(real(c)) && imag(c) == 0);
assert(real(d) == Inf && imag(d) == 0);
a = mt(y, 3); b = mrd(y, 3); c = mrd(3, y); d = rdv(y, 3);
assert(isnan(real(a)) && imag(a) == 0 && isnan(real(c)) && imag(c) == 0);
assert(isnan(real(b)) && imag(b) == 0 && isnan(real(d)) && imag(d) == 0);
a = mt(Inf, 1i); b = mt(1i, Inf); c = mrd(3i, 0); d = mt(y, 2 + 0i);
assert(real(a) == 0 && imag(a) == Inf && real(b) == 0 && imag(b) == Inf);
assert(real(c) == 0 && imag(c) == Inf);
assert(isnan(real(d)) && imag(d) == 0);

% genuinely complex: a part stays 0 where the complex operand's is 0
a = mt(2i, NaN); b = mt(1 + 2i, Inf); c = mrd(1 + 2i, 0); d = mrd(2i, NaN);
assert(real(a) == 0 && isnan(imag(a)));
assert(real(b) == Inf && imag(b) == Inf);
assert(real(c) == Inf && imag(c) == Inf);
assert(real(d) == 0 && isnan(imag(d)));
a = mt(1 + 2i, 3 + 4i); b = mrd(2, 1 + 1i); c = mrd(-5 + 10i, 3 + 4i);
assert(real(a) == -5 && imag(a) == 10);
assert(real(b) == 1 && imag(b) == -1);
assert(abs(c - (1 + 2i)) < 1e-15);

disp('SUCCESS')

function y = mt(a, b)
  %!numbl:assert_jit c
  y = a * b;
end

function y = mrd(a, b)
  %!numbl:assert_jit c
  y = a / b;
end

function y = rdv(a, b)
  %!numbl:assert_jit c
  y = a ./ b;
end
