% DIAGNOSIS: NaN in a scalar `if`/`while` condition. The interpreter's toBool
%   returns (v !== 0), so NaN is TRUE (matches MATLAB). The JS-JIT emits the raw
%   `if (x)` and JS treats NaN as falsy, so opt1 skips the branch.
%   (convert.ts toBool vs emitJs truthy()/raw expr.)
% --opt 0 output: 200
% --opt 1 output: 0
c=0; for k=1:200; x=0/0; if x; c=c+1; end; end; disp(c)
