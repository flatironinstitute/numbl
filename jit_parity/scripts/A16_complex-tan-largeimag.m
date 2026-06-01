% DIAGNOSIS: complex tan() diverges for large imaginary parts. The
% interpreter (opt0) uses the double-angle formula
%   tan = [sin(2re) + i*sinh(2im)] / [cos(2re) + cosh(2im)]
% which preserves the tiny real part; the JS-JIT runtime mtoc2_ctan
% computes sin(z)/cos(z) with a naive denominator c.re^2+c.im^2 that
% loses the real part to 0 (and overflows to NaN sooner).
% opt0: 3.4829e-174 + 1i
% opt1: 1i
function s = f(z, n)
  s = z;
  for k=1:n
    s = tan(z);
  end
end
disp(f(complex(1, 200), 500))
