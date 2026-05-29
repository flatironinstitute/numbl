% Edge cases that involve a tensor binding being read and written within
% the same statement: the prior wrapper must not be released before the
% RHS finishes evaluating.

%!numbl:assert_jit
x = [10, 20, 30, 40, 50];

% RHS is the same variable. Identity assignment must not corrupt x.
x = x;
assert(isequal(x, [10, 20, 30, 40, 50]), 'x = x should be a no-op');

% RHS uses x to compute a new tensor. The old x must remain readable for
% the entire expression evaluation.
x = x + 1;
assert(isequal(x, [11, 21, 31, 41, 51]), 'x = x + 1 should add 1 to each');

x = x .* x;
assert(isequal(x, [121, 441, 961, 1681, 2601]), 'x = x .* x should square');

% Self-aliasing through a function call.
y = [1, 2, 3];
y = passthrough(y);
assert(isequal(y, [1, 2, 3]), 'passthrough should be identity');

% Multi-output where one output is the input.
v = [5, 10, 15];
[a, b] = pair(v);
assert(isequal(a, [5, 10, 15]) && isequal(b, [25, 100, 225]), ...
  'pair should return v and v.^2');
assert(isequal(v, [5, 10, 15]), 'caller v unchanged');

% Index-then-reassign-whole-tensor in one step. Index store returns the
% mutated wrapper; the env.set release must not double-release after the
% (no-op) COW.
w = (1:10);
w(5) = 99;
assert(w(5) == 99 && w(1) == 1 && w(10) == 10);

% Repeated identity assignment in a loop.
v2 = ones(1, 100);
for k = 1:100
  v2 = v2;  %#ok<ASGSL>
end
assert(isequal(v2, ones(1, 100)), 'identity loop should preserve v2');

disp('SUCCESS')

function y = passthrough(x)
  y = x;
end

function [a, b] = pair(v)
  a = v;
  b = v .* v;
end
