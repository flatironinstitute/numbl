% DIAGNOSIS: complex power with a zero base diverges. The interpreter
% (opt0) complexPow returns 0 whenever Re(exp)>0 (any imag), and
% Infinity for a negative/zero-real exponent. The JS-JIT runtime
% mtoc2_cpow only returns 0 when the exponent is real-positive
% (b.im===0) and returns NaN+NaNi in every other zero-base case.
% opt0: 0          (for 0 ^ (2+3i))
% opt1: NaN + NaNi
% Also: 0 ^ (-2)    -> opt0 Infinity, opt1 NaN+NaNi
%       0 ^ (0+2i)  -> opt0 Infinity, opt1 NaN+NaNi
function s = f(a, b, n)
  s = a;
  for k=1:n
    s = a ^ b;
  end
end
disp(f(complex(0,0), complex(2,3), 500))
