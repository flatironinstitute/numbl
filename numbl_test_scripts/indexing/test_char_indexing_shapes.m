% Char arrays: read shapes, growth padding with char(0), deletion.

x = 'abc';
assert(isequal(size(x([])), [0 0]) && ischar(x([])));
assert(isequal(size(x(zeros(1, 0))), [1 0]) && ischar(x(zeros(1, 0))));
assert(isequal(size(x(logical([0 0 0]))), [1 0]));
assert(isequal(x([1; 2]), 'ab'));
assert(isequal(size(x(:)), [3 1]) && isequal(x(:)', 'abc'));
assert(isequal(x([3 1 2]), 'cab'));
assert(isequal(x(logical([1 0 1])), 'ac'));
assert(isequal(x(end:-1:1), 'cba'));

M = ['abc'; 'def'];
assert(isequal(M(:, 2), ['b'; 'e']));
assert(isequal(M(2, :), 'def'));
assert(isequal(M([1 2 3]), 'adb'));
assert(isequal(M(:)', 'adbecf'));
assert(isequal(M(:, [3 1]), ['ca'; 'fd']));
assert(isequal(M([2 1], :), ['def'; 'abc']));

% Growth pads with char(0)
y = 'abc'; y(5) = 'e'; assert(ischar(y) && isequal(double(y), [97 98 99 0 101]));
y = 'abc'; y(2, :) = 'def'; assert(isequal(y, ['abc'; 'def']));
y = ['abc'; 'def']; y(3, 1) = 'z';
assert(isequal(size(y), [3 3]) && isequal(double(y(3, :)), [122 0 0]));
y = ''; y(3) = 'a'; assert(isequal(double(y), [0 0 97]));
y = 'abc'; y(2) = 66; assert(isequal(y, 'aBc'));

% Deletion
y = 'abc'; y([]) = []; assert(isequal(y, 'abc'));
y = 'abc'; y(2) = []; assert(isequal(y, 'ac'));
y = ['abc'; 'def']; y(:, 2) = []; assert(isequal(y, ['ac'; 'df']));

try
    x(4); error('no error');
catch e
    assert(strcmp(e.message, 'Index exceeds the number of array elements. Index must not exceed 3.'));
end

disp('SUCCESS');
