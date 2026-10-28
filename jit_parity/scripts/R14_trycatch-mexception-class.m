% R: `catch ME` binds an MException object for errors raised by JIT'd
% code: its class, identifier, message, stack and report agree across
% modes, as does passing the exception into a function that reads it.
for n = [2 5]
  try
    disp(sumfirst([4 5 6], n));
  catch ME
    fprintf('%s %d [%s] [%s] %s:%d\n', class(ME), isa(ME, 'MException'), ...
      ME.identifier, ME.message, ME.stack(1).name, ME.stack(1).line);
    fprintf('%s\n', getReport(ME, 'basic'));
    fprintf('%s\n', describe(ME));
  end
end
for x = [1 -1]
  try
    disp(checked_twice(x));
  catch ME
    fprintf('%s [%s] [%s] %s\n', class(ME), ME.identifier, ME.message, ...
      ME.stack(1).name);
    try
      rethrow(ME);
    catch ME2
      fprintf('rethrown: [%s] %s:%d\n', ME2.identifier, ME2.stack(1).name, ...
        ME2.stack(1).line);
    end
  end
end

function s = sumfirst(v, n)
  s = 0;
  for k = 1:n
    s = s + v(k);
  end
end

function r = checked_twice(x)
  r = 0;
  for k = 1:3
    if x < 0
      error('parity:neg', 'negative input %g at k=%d', x, k);
    end
    r = r + 2 * x;
  end
end

function s = describe(ME)
  s = [ME.identifier '|' ME.message];
end
