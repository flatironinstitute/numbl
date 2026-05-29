% A tensor passed through a chain of function calls must be unaffected by
% mutations inside any of the callees, and the caller's buffer must stay
% alive across all the calls.

%!numbl:assert_jit
v = (1:100)';
result = level1(v);
assert(isequal(v, (1:100)'), 'caller v must survive deep call chain');
assert(result == sum((1:100).^2 + 1), sprintf('expected %d, got %d', ...
  sum((1:100).^2 + 1), result));

% Repeat in a loop — the COW invariant must hold for many calls.
total = 0;
for k = 1:30
  total = total + level1(v);
end
assert(isequal(v, (1:100)'), 'caller v must survive loop of calls');
assert(total == 30 * sum((1:100).^2 + 1), 'aggregate sum should be correct');

% Pass the same tensor twice in one call — both args must independently COW.
[a, b] = double_arg(v);
assert(isequal(v, (1:100)'), 'caller v must survive double-arg call');
assert(a(1) == 999 && a(2) == 2, 'first arg copy got the mutation');
assert(b(1) == 1 && b(2) == 888, 'second arg copy got its own mutation');

disp('SUCCESS')

function r = level1(v)
  v(1) = -1;
  r = level2(v);
end

function r = level2(v)
  v(2) = -2;
  r = level3(v);
end

function r = level3(v)
  v(3) = -3;
  v = v.^2 + 1;
  r = sum(v);
end

function [a, b] = double_arg(x, y)
  if nargin < 2
    y = x;
  end
  x(1) = 999;
  y(2) = 888;
  a = x;
  b = y;
end
