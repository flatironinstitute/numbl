% Cell arrays: paren/brace reads, growth padding with [], deletion, 3-D.

c = {1, 2, 3};
r = {1; 2; 3};
m = {1, 2; 3, 4};

% Paren reads follow the numeric shape rules
assert(isequal(size(c([1; 2])), [1 2]));
assert(isequal(size(r([1 2])), [2 1]));
assert(isequal(m([1 2]), {1, 3}));
assert(isequal(m([1; 2]), {1; 3}));
assert(isequal(m(:), {1; 3; 2; 4}));
assert(isequal(m(logical([1 0 1 1])), {1, 2, 4}));
assert(isequal(m(logical([1 0 1])), {1, 2}));
assert(isequal(size(c([])), [0 0]));
assert(isequal(size(c(zeros(1, 0))), [1 0]));
assert(isequal(size(c(logical([0 0 0]))), [1 0]));
assert(isequal(m(:, 2), {2; 4}));
assert(isequal(m(2, :), {3, 4}));
assert(isequal(m([1 2; 2 1]), {1, 3; 3, 1}));
assert(isequal(c(1, 1, 1), {1}));
assert(isequal(c(:, :, 1), c));

% Brace reads
assert(m{2, 1} == 3);
[p, q] = m{:, 2};
assert(p == 2 && q == 4);
[p, q] = c{logical([1 0 1])};
assert(p == 1 && q == 3);

% Growth pads with []
d = {}; d(3) = {5}; assert(isequal(size(d), [1 3]) && isempty(d{1}) && d{3} == 5);
d = {}; d{3} = 5; assert(isequal(size(d), [1 3]));
d = cell(0, 1); d{3} = 5; assert(isequal(size(d), [3 1]));
d = {1; 2}; d{4} = 5; assert(isequal(size(d), [4 1]));
d = {1, 2; 3, 4}; d{3, 3} = 5; assert(isequal(size(d), [3 3]) && isempty(d{3, 1}) && d{3, 3} == 5);
d = {1, 2, 3}; d{2, 1} = 1; assert(isequal(size(d), [2 3]) && isempty(d{2, 2}));
d = {}; d(:, 1) = {1; 2}; assert(isequal(d, {1; 2}));
d = {}; d(2, :) = {1, 2}; assert(isequal(size(d), [2 2]) && d{2, 2} == 2);
d = {1, 2}; d(2, :) = {3, 4}; assert(isequal(d, {1, 2; 3, 4}));
try
    d = {1, 2; 3, 4}; d{5} = 5; error('no error');
catch e
    assert(strcmp(e.message, 'Attempt to grow array along ambiguous dimension.'));
end

% Paren assignment needs a cell RHS; 1x1 cells broadcast
d = {1, 2, 3}; d([1 2]) = {9}; assert(isequal(d, {9, 9, 3}));
d = {1, 2, 3}; d([1 2]) = {8, 9}; assert(isequal(d, {8, 9, 3}));
d = {1, 2, 3}; d(:) = {0}; assert(isequal(d, {0, 0, 0}));
d = {1, 2, 3}; d(2) = {[]}; assert(isequal(size(d), [1 3]) && isempty(d{2}));
d = {1, 2, 3}; d{2} = []; assert(isequal(size(d), [1 3]) && isempty(d{2}));
d = {1, 2, 3}; d([2 1]) = d(1:2); assert(isequal(d, {2, 1, 3}));
try
    d = {1, 2, 3}; d([1 2]) = {8, 9, 10}; error('no error');
catch e
    assert(strcmp(e.message, 'Unable to perform assignment because the left and right sides have a different number of elements.'));
end
try
    d = {1, 2, 3}; d(2) = 5; error('no error');
catch e
    assert(strcmp(e.message, 'Conversion to cell from double is not possible.'));
end
try
    x = []; x(3) = {5}; error('no error');
catch e
    assert(strcmp(e.message, 'Conversion to double from cell is not possible.'));
end

% Deletion
d = {1, 2, 3}; d(2) = []; assert(isequal(d, {1, 3}));
d = {1; 2; 3}; d(2) = []; assert(isequal(d, {1; 3}));
d = {1, 2; 3, 4}; d(2) = []; assert(isequal(d, {1, 2, 4}));
d = {1, 2; 3, 4}; d(:, 1) = []; assert(isequal(d, {2; 4}));
d = {1, 2; 3, 4}; d(1, :) = []; assert(isequal(d, {3, 4}));

% 3-D cells
e3 = cell(2, 2, 2);
assert(isequal(size(e3(:, :, 2)), [2 2]));
e3{2, 2, 2} = 7;
assert(isequal(e3(2, 2, 2), {7}));
e3(:, :, 1) = [];
assert(isequal(size(e3), [2 2]) && e3{2, 2} == 7);
d = {1, 2, 3}; d(1, 1, 2) = {7};
assert(isequal(size(d), [1 3 2]) && d{1, 1, 2} == 7 && isempty(d{1, 2, 2}));

% Copy-on-write
a = {1, 2}; b = a; b{1} = 9; assert(a{1} == 1 && b{1} == 9);
a = {1, 2}; b = a; b(4) = {9}; assert(numel(a) == 2 && numel(b) == 4);

try
    c(4); error('no error');
catch e
    assert(strcmp(e.message, 'Index exceeds the number of array elements. Index must not exceed 3.'));
end

disp('SUCCESS');
