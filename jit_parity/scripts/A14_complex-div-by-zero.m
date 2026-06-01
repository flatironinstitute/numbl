% DIAGNOSIS: complex division by an exactly-zero divisor diverges. The
% interpreter (opt0) complexDivide() special-cases a zero divisor and
% returns signed Inf (nonzero numerator) or NaN (0/0). The JS-JIT
% runtime mtoc2_cdiv has no zero-divisor branch: Smith's algorithm
% yields 0/0 = NaN in both components.
% opt0: Infinity + Infinityi
% opt1: NaN + NaNi
function s = f(a, b, n)
  s = a;
  for k=1:n
    s = a / b;
  end
end
disp(f(complex(3,4), complex(0,0), 500))
