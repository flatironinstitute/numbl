% Diagnosis: opt0 routes any/all of a row vector through the native
%   -ffast-math LAPACK addon (scanLogical fast path via realFlatReduce),
%   which mishandles NaN: all([1 NaN 0]) returns 1 even though a literal 0
%   is present. opt1 (JS-JIT logical_all) follows MATLAB and returns 0.
%   any([0 NaN 0]) similarly: opt0=0, opt1=1.
% --opt 0 output: 1
% --opt 1 output: 0
function r=f(v)
  s=false;
  for k=1:300
    s=all(v);
  end
  r=s;
end
disp(f([1 NaN 0]))
