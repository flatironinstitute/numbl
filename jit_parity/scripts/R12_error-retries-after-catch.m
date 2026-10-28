% R: a runtime error does not permanently de-JIT a unit. The same JIT'd
% function errors for some inputs and succeeds for others; every call
% gives the interpreter's result.
v = [5 6 7];
for n = [2 4 3 5 1]
  try
    fprintf('n=%d sum=%g\n', n, sumfirst(v, n));
  catch ME
    fprintf('n=%d [%s]\n', n, ME.message);
  end
end

function s = sumfirst(v, n)
  s = 0;
  for k = 1:n
    s = s + v(k);
  end
end
