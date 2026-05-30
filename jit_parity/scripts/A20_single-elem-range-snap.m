% DIAGNOSIS: Single-element range snaps to `end` in JS-JIT but not in interpreter.
%   makeRangeTensor (interp) only snap-to-end when n>1 (tensor-construction.ts:52);
%   mtoc2_range_value (JIT) snaps whenever i===count-1, incl. the count==1 case.
%   For 0:1000:1e-9 the range has exactly ONE element: interp value = start (0),
%   JIT value = end (1e-9).
% --opt 0 output: 0
% --opt 1 output: 1e-9
x=-1; for rep=1:200; for k=0:1000:1e-9; x=k; end; end; disp(x)
