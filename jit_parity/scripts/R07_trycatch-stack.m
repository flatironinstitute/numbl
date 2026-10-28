% R: an error raised directly by a JIT'd function carries the same
% ME.stack (function name and line) as the interpreter's.
for n = [2 5]
  try
    disp(firstn([4 5 6], n));
  catch ME
    fprintf('[%s] [%s] %s:%d\n', ME.message, ME.identifier, ...
      ME.stack(1).name, ME.stack(1).line);
  end
end

function s = firstn(v, n)
  s = 0;
  for k = 1:n
    s = s + ...
      v(k);
  end
end
