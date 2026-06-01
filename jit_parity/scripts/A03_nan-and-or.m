% DIAGNOSIS: scalar and()/or()/&&/|| with a runtime NaN operand diverge.
% The JIT (_shortcircuit.ts emitJs) coerces operands with `!!(x)`, and
% `!!NaN === false` in JS, so NaN counts as false. The interpreter coerces
% via `n !== 0`, where `NaN !== 0` is true, so NaN counts as true.
% Root cause: src/numbl-core/jit/builtins/defs/logical/_shortcircuit.ts
% emitJs uses `!!(...)`; backs and / or / && / ||.
% opt0 output (2 lines): 1 1
% opt1 output (2 lines): 1 0   (and(NaN,1): 1 vs 0)
function r = opf(a, b)
  r = and(a, b);
end
A = [1 NaN];
B = [1 1];
out = zeros(1, 2);
for k=1:1000
  idx = mod(k-1, 2) + 1;
  out(idx) = opf(A(idx), B(idx));
end
for j=1:2
  fprintf('%g\n', out(j));
end
