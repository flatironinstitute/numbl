% Test that element deletion on empty vectors preserves shape

% Empty column vector: x(logical_idx) = [] should stay 0x1
x = zeros(0,1);
x(isnan(x)) = [];
assert(size(x, 1) == 0);
assert(size(x, 2) == 1);

% Empty row vector: x(logical_idx) = [] should stay 1x0
y = zeros(1,0);
y(isnan(y)) = [];
assert(size(y, 1) == 1);
assert(size(y, 2) == 0);

% Non-empty column: delete all elements, stays column (0x1)
z = [1;2;3];
z([true;true;true]) = [];
assert(size(z, 1) == 0);
assert(size(z, 2) == 1);

% Non-empty column: delete some elements, stays column
w = [1;2;3;4];
w([true;false;true;false]) = [];
assert(size(w, 1) == 2);
assert(size(w, 2) == 1);
assert(isequal(w, [2;4]));

% Chebfun pattern: sort, isnan delete, then vertcat with false
x = zeros(0,1);
y = sort(x(:));
y(isnan(y)) = [];
assert(size(y, 1) == 0);
assert(size(y, 2) == 1);

disp('SUCCESS');
