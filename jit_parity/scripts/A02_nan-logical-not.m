% DIAGNOSIS: (same NaN root cause as 03, distinct emit site) `~x` for x=NaN.
%   Interp not(): (x===0)?1:0 -> ~NaN = 0. JS-JIT not.ts emits `!(x)`; JS
%   `!NaN` === true -> 1. Also affects `&&`/`||` which emit `!!(x)`.
% --opt 0 output: 0
% --opt 1 output: 200
c=0; for k=1:200; x=0/0; if ~x; c=c+1; end; end; disp(c)
