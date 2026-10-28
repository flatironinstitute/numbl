% R: an error deep in a recursive JIT'd function. Every recursive call is
% an interpreter frame, which the JIT does not track, so the unit re-runs
% on the interpreter to report the full stack.
disp(down([1 2 3], 3));
disp(down([1 2 3], 4));

function s = down(v, k)
  if k == 0
    s = 0;
  else
    s = v(k) + down(v, k - 1);
  end
end
