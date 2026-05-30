% Unary `~` (logical not) must be able to start a whitespace-separated matrix
% or cell element, not just the first one. numbl's matrix/cell continuation
% `canStart` set included Plus/Minus but omitted Tilde, so `[1 ~0]` failed to
% parse.

assert(isequal([1 ~0], [1 1]), '~ as second matrix element');
assert(isequal([~0 ~1], [1 0]), '~ as both elements');
assert(isequal([1 ~0 3], [1 1 3]), '~ in the middle');

x = 5;
assert(isequal([1 ~x], [1 0]), '~ of a variable');

% cell array too
c = {1, ~0};
assert(isequal(c{2}, true), '~ in cell element');

disp('SUCCESS');
