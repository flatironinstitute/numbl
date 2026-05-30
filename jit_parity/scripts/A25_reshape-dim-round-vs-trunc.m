% AREA: shape/structural builtins — reshape() dynamic non-integer dim
%
% WHAT IT TESTS: reshape(x, n, []) with a runtime non-integer dim
% (n=2.6, read from a tensor element so it is not folded) and an auto
% `[]` slot. The interpreter ROUNDS the dim; the JIT (both JS and C)
% TRUNCATES it. With 6 elements both shapes are legal, so the result is
% a SILENTLY DIFFERENT shape/value (no error).
%
% x=[1 2 3 4 5 6]; f returns r(1,2) of reshape(x,n,[]):
%   opt0 (interp, REFERENCE): 4   (round(2.6)=3 -> 3x2, r(1,2)=4)
%   opt1 (JS-JIT):            3   (trunc(2.6)=2 -> 2x3, r(1,2)=3)   <-- DIVERGES
%   opt2 (C-JIT):             3   (same trunc)                      <-- DIVERGES
%
% HYPOTHESIZED CAUSE: src/numbl-core/jit/builtins/defs/shape/reshape.ts
% dimC() emits `(long)(arg)` and emitJs/call use `Math.trunc(arg)` for a
% dynamic dim, but the interpreter (array-manipulation.ts, reshape) uses
% `Math.round(toNumber(a))`. JIT should round to match the reference.
%
% JIT ENGAGEMENT: CONFIRMED. opt2 --dump-c (941 lines) call site is
%   mtoc2_reshape_nd(x, 2, (long[]){(long)(n), -1L}).
function s = f(vals, k, x)
  n = vals(k);
  r = reshape(x, n, []);
  s = r(1, 2);
end
vals = [2.6 2.6 2.6 2.6 2.6];
x = [1 2 3 4 5 6];
total = -1;
for k = 1:5
  total = f(vals, k, x);
end
disp(total);
