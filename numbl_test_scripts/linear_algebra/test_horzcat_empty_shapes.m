% Test horzcat with empty arrays of various shapes

% horzcat of [0x1] with [1x3] should work (MATLAB drops zero-element arrays)
z = zeros(0,1);
b = [1 2 3];
r = [z b];
assert(isequal(size(r), [1 3]), 'horzcat [0x1] [1x3] shape');
assert(isequal(r, [1 2 3]), 'horzcat [0x1] [1x3] values');

% horzcat of [1x0] with [1x3]
z2 = zeros(1,0);
r2 = [z2 b];
assert(isequal(size(r2), [1 3]), 'horzcat [1x0] [1x3] shape');
assert(isequal(r2, [1 2 3]), 'horzcat [1x0] [1x3] values');

% horzcat of [0x0] with [1x3]
r3 = [[] b];
assert(isequal(size(r3), [1 3]), 'horzcat [0x0] [1x3] shape');

% Column vector logical indexing producing [0x1], then horzcat
y = [1; 2; 3];
empty_col = y(y > 10);  % should be [0x1]
assert(isequal(size(empty_col), [0 1]), 'col logical indexing all-false shape');
r4 = [empty_col' b];  % transpose to row first should work
assert(isequal(r4, [1 2 3]), 'horzcat transposed empty with row');

% Direct horzcat of [0x1] with row - MATLAB allows this
r5 = [empty_col b];
assert(isequal(size(r5), [1 3]), 'horzcat [0x1] directly with [1x3]');
assert(isequal(r5, [1 2 3]), 'horzcat [0x1] directly with [1x3] values');

disp('SUCCESS');
