% Test cell array parenthesis indexing returns cell (not contents)

c = {10, 'hello', [1 2 3]};

% Scalar paren index should return 1x1 cell
r1 = c(1);
assert(iscell(r1), 'c(1) should be cell');
assert(isequal(size(r1), [1 1]), 'c(1) size should be 1x1');
assert(r1{1} == 10, 'c(1){1} contents');

% Curly brace should still return contents directly
r2 = c{1};
assert(r2 == 10, 'c{1} returns contents');
assert(~iscell(r2), 'c{1} is not cell');

% Range paren index should return 1x2 cell
r3 = c(1:2);
assert(iscell(r3), 'c(1:2) should be cell');
assert(isequal(size(r3), [1 2]), 'c(1:2) size should be 1x2');

% 2D cell paren indexing
c2 = {1 2; 3 4};
r4 = c2(1, 1);
assert(iscell(r4), 'c2(1,1) should be cell');

r5 = c2(2, 2);
assert(iscell(r5), 'c2(2,2) should be cell');
assert(r5{1} == 4, 'c2(2,2){1} contents');

disp('SUCCESS');
