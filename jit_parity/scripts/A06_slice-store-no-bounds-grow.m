% DIAGNOSIS: range-slice indexed STORE that grows the array
%   (the canonical `v(end+1:end+k) = ...` append) is NOT bounds-checked
%   in the JS-JIT. Writes past the Float64Array end are silently dropped
%   (a no-op in JS), so the array never grows. The interpreter grows it.
%   Root cause: emitJs.ts emitIndexSliceStoreJs single-slot Range path
%   emits no bounds/grow check (the C path uses mtoc2_check_linear_range,
%   and scalar IndexStore uses grow-bail; slice store has neither).
% --opt 0 output: 6030   (v becomes [1 2 3 7 8 9], sum=30, numel=6)
% --opt 1 output: 3006   (v stays [1 2 3], OOB writes dropped, numel=3)
function r = f(v)
  v(end+1:end+3) = [7 8 9];
  r = sum(v) + numel(v)*1000;
end
x = [1 2 3];
disp(f(x));
