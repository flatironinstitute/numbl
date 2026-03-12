% Array slicing and indexing

v = [10, 20, 30, 40, 50];

% Range indexing
sub = v(2:4);
assert(length(sub) == 3)
assert(sub(1) == 20)
assert(sub(2) == 30)
assert(sub(3) == 40)

% End keyword
last = v(end);
assert(last == 50)

sub2 = v(3:end);
assert(length(sub2) == 3)
assert(sub2(1) == 30)

% 2D matrix slicing
A = [1,2,3; 4,5,6; 7,8,9];

% Row slice
row2 = A(2, :);
assert(length(row2) == 3)
assert(row2(1) == 4)
assert(row2(3) == 6)

% Column slice
col3 = A(:, 3);
assert(length(col3) == 3)
assert(col3(1) == 3)
assert(col3(3) == 9)

% Submatrix
sub = A(1:2, 2:3);
assert(size(sub, 1) == 2)
assert(size(sub, 2) == 2)
assert(sub(1,1) == 2)
assert(sub(2,2) == 6)

disp('SUCCESS')
