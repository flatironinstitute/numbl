% Test calling a function with fewer output arguments than declared.
% In MATLAB, you can request fewer outputs than a function declares.

% Function declares 4 outputs, call with 2
[a, b] = four_outputs(10);
assert(a == 10);
assert(b == 20);

% Call with 1
c = four_outputs(5);
assert(c == 5);

% Call with 3
[x, y, z] = four_outputs(7);
assert(x == 7);
assert(y == 14);
assert(z == 21);

disp('SUCCESS')

function [a, b, c, d] = four_outputs(x)
  a = x;
  b = 2 * x;
  c = 3 * x;
  d = 4 * x;
end
