% AREA: shape/structural builtins — repmat dynamic (non-integer) rep arg
%
% WHAT IT TESTS: repmat with a runtime, non-integer replication count
% (2.6, read from a tensor indexed by the loop var so it is NOT folded).
% The interpreter and JS-JIT ROUND the rep; the C-JIT TRUNCATES it.
%
% f returns numel(repmat([1 2 3], 1, n)) with n=2.6:
%   opt0 (interp, REFERENCE): 9   (round(2.6)=3 tiles -> 3*3)
%   opt1 (JS-JIT):            9   (Math.round(n)=3)
%   opt2 (C-JIT):             6   (long)2.6=2 tiles -> 3*2     <-- DIVERGES
%
% HYPOTHESIZED CAUSE: src/numbl-core/jit/builtins/defs/shape/repmat.ts
% repC() emits `(long)(<arg>)` (truncation) for a dynamic rep, while
% repJs() emits `Math.round(<arg>)` and the interpreter
% (array-manipulation.ts) also uses Math.round. C should round to match.
%
% JIT ENGAGEMENT: CONFIRMED. opt2 --dump-c (877 lines) call site is
%   mtoc2_tensor_repmat(_mtoc2_t1, 2, (long[]){1L, (long)(n)})
% opt1 --dump-js call site is mtoc2_tensor_repmat(..., [1, Math.round(n)]).
function s = f(vals, k)
  n = vals(k);
  r = repmat([1 2 3], 1, n);
  s = numel(r);
end
vals = [2.6 2.6 2.6 2.6 2.6];
total = -1;
for k = 1:5
  total = f(vals, k);
end
disp(total);
