% `clear x` and similar paths route through Environment.delete and
% clearLocals, both of which release prior tensor bindings. The variable
% must be fully gone afterward (treated as undefined on next access).

x = [1, 2, 3, 4, 5];
clear x;
assert(~exist('x', 'var'), 'x should not exist after clear');

% Multiple clears in one statement.
a = (1:10);
b = (1:20);
c = (1:30);
clear a b c;
assert(~exist('a', 'var') && ~exist('b', 'var') && ~exist('c', 'var'), ...
  'a, b, c should all be cleared');

% Reassign after clear.
y = [10, 20, 30];
clear y;
y = [99];
assert(isequal(y, 99), 'y should be 99 after re-assign');

% Mixed clear + reassign in a loop.
for i = 1:50
  m = (1:100) * i;
  clear m;
  m = ones(1, 50);
  assert(length(m) == 50);
end

disp('SUCCESS')
