% R: an error raised in a callee of the JIT'd unit (here a function the
% loop calls). The JIT does not track the callee's frame, so the unit
% re-runs on the interpreter, which reports the full call stack.
total = 0;
for k = 1:4
  total = total + pick([10 20 30], k);
  fprintf('total=%g\n', total);
end

function y = pick(v, k)
  y = v(k);
end
