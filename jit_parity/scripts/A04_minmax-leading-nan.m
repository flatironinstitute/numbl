% Diagnosis: JIT runtime minmax_all() has NO NaN guard, so a LEADING NaN
%   poisons the running accumulator (best=NaN; x>NaN is always false).
%   The interpreter follows MATLAB and ignores NaN in min/max.
% --opt 0 output: 3
% --opt 1 output: NaN
function r=f(v)
  s=0;
  for k=1:300
    s=max(v);
  end
  r=s;
end
disp(f([NaN 1 3]))
