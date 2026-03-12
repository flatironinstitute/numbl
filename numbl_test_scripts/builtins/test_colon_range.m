% Test colon operator for range generation (array.ts coverage)

% Basic 2-arg colon
x = 1:5;
assert(isequal(x, [1 2 3 4 5]));

% 3-arg colon with step
x = 1:2:9;
assert(isequal(x, [1 3 5 7 9]));

% Descending range
x = 5:-1:1;
assert(isequal(x, [5 4 3 2 1]));

% Fractional step
x = 0:0.5:2;
assert(isequal(x, [0 0.5 1 1.5 2]));

% Single element range
x = 5:5;
assert(isequal(x, 5));

% Empty range (start > stop with positive step)
x = 5:1;
assert(isempty(x));

disp('SUCCESS');
