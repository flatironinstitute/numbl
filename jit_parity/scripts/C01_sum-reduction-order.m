% Diagnosis: opt0 sums via the native -ffast-math LAPACK addon, whose
%   scalar accumulation loop gcc auto-vectorizes (SIMD lanes + horizontal
%   add) => a DIFFERENT addition order than opt1's strictly-sequential JS
%   reducer (acc += x). For ill-conditioned data the rounded result
%   diverges. Affects sum and mean (and prod). E.g. repmat(0.1,1,100) also
%   diverges in the low bits: opt0=10.000000000000003553 vs
%   opt1=9.9999999999999804601.
% --opt 0 output: 48
% --opt 1 output: 0
function r=f(v)
  s=0;
  for k=1:300
    s=sum(v);
  end
  r=s;
end
v = [1e16, ones(1,64), -1e16];
fprintf('%.20g\n', f(v))
