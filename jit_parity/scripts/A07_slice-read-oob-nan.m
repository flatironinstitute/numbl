% DIAGNOSIS: multi-slot per-axis READ (e.g. M(3,:)) with an out-of-bounds
%   axis index. The single-slot Range read DOES bounds-check, but the
%   multi-slot per-axis read path in emitJs.ts (emitIndexSliceJs, the
%   ndim>1 branch) emits NO per-axis bounds check. OOB reads return
%   undefined from the Float64Array -> NaN, while the interpreter errors.
% --opt 0 output: ERROR "Index exceeds array bounds" (exit 1)
% --opt 1 output: NaN  (reads past buffer, exit 0)
function r = f(M)
  s = 0;
  for k=1:200
    w = M(3,:);
    s = sum(w);
  end
  r = s;
end
M = [1 2 3; 4 5 6];
disp(f(M));
