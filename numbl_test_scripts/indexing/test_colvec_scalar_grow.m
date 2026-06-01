% Growing a vector via a scalar linear index preserves its orientation.
% MATLAB: a column vector stays a column; a row stays a row.

a = [1; 2; 3]; a(5) = 9;
assert(isequal(size(a), [5 1]));
assert(isequal(a, [1; 2; 3; 0; 9]));

a = zeros(3, 1); a(4) = 1;
assert(isequal(size(a), [4 1]));

% row vector stays a row
b = [1 2 3]; b(5) = 9;
assert(isequal(size(b), [1 5]));
assert(isequal(b, [1 2 3 0 9]));

% empty grows to a row
c = []; c(3) = 7;
assert(isequal(size(c), [1 3]));

disp('SUCCESS')
