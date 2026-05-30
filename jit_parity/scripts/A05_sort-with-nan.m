% Diagnosis: JIT sort comparator (pair_sort_indices) returns the stable
%   tie-break (p-q) for any NaN comparison since NaN<x and NaN>x are both
%   false. This makes the comparator non-transitive, so V8's sort leaves
%   NaN-containing arrays effectively unsorted. The interpreter places NaN
%   at the end (MATLAB semantics) and sorts the rest.
% --opt 0 output:    1   2   3   NaN
% --opt 1 output:    3   NaN   1   2
function r=f(v)
  s=v;
  for k=1:300
    s=sort(v);
  end
  r=s;
end
w=f([3 NaN 1 2]);
disp(w)
