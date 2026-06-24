% MATLAB allows function output arguments separated by spaces (not just
% commas): `function [a b] = f()`.
[p, q] = space_out_helper_(5);
assert(p == 5);
assert(q == 6);
disp('SUCCESS')

function [a b] = space_out_helper_(x)
  a = x;
  b = x + 1;
end
