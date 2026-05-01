% Functions that throw mid-body must still leave the caller's tensors intact.
% The catch path runs clearLocals; we verify the caller's variables are
% unaffected and re-callable.

v = [1, 2, 3, 4, 5];

for k = 1:10
  try
    bad_function(v);
    error('expected throw');
  catch err
    assert(strcmp(err.message, 'oops') || contains(err.message, 'oops'), ...
      sprintf('unexpected error: %s', err.message));
  end
end
assert(isequal(v, [1, 2, 3, 4, 5]), 'caller v must be unchanged after errors');

% Function that mutates then errors — caller's data still intact.
for k = 1:10
  try
    mutate_then_throw(v);
  catch
    % swallow
  end
end
assert(isequal(v, [1, 2, 3, 4, 5]), 'caller v must be unchanged after mutate-then-throw');

% Inner function throws, outer catches, returns a valid tensor.
out = catch_and_return(v);
assert(isequal(out, [1, 2, 3, 4, 5]), 'returned tensor should match v');
assert(isequal(v, [1, 2, 3, 4, 5]), 'caller v unchanged through try/catch');

disp('SUCCESS')

function bad_function(~)
  error('oops');
end

function mutate_then_throw(v)
  v(1) = 999;
  error('after mutation');
end

function out = catch_and_return(v)
  try
    bad_function(v);
  catch
    % ignore
  end
  out = v;
end
