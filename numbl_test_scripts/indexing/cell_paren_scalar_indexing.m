% Test cell paren indexing with scalar indices
% In MATLAB, c(i) and c(i,j) return cells, not the contents
% c{i} and c{i,j} return the contents

% Test 1D linear indexing on 2D cell array
c = {1, 2; 3, 4; 5, 6};  % 3x2 cell
assert(isequal(c(1), {1}), 'c(1) should return cell containing 1');
assert(isequal(c(2), {3}), 'c(2) should return cell containing 3');
assert(isequal(c(3), {5}), 'c(3) should return cell containing 5');
assert(isequal(c(4), {2}), 'c(4) should return cell containing 2');

% Test 2D scalar indexing
c = {1, 2; 3, 4};
assert(isequal(c(1, 1), {1}), 'c(1,1) should return cell');
assert(isequal(c(1, 2), {2}), 'c(1,2) should return cell');
assert(isequal(c(2, 1), {3}), 'c(2,1) should return cell');
assert(isequal(c(2, 2), {4}), 'c(2,2) should return cell');

% Test cell braces returns contents
assert(isequal(c{1, 1}, 1), 'c{1,1} should return contents');
assert(isequal(c{2, 2}, 4), 'c{2,2} should return contents');

% Test mixed vector and scalar
assert(isequal(c([1 2], 1), {1; 3}), 'c([1 2], 1) should return cell vector');
assert(isequal(c(1, [1 2]), {1, 2}), 'c(1, [1 2]) should return cell vector');

disp('SUCCESS');
