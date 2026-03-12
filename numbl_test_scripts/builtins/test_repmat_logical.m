% Test that repmat works with logical values

% Test 1: repmat a scalar logical
a = true;
r = repmat(a, 1, 3);
assert(isequal(r, [true, true, true]), 'repmat(true, 1, 3) should be [true true true]');
assert(islogical(r), 'result should be logical');

% Test 2: repmat false
b = false;
r2 = repmat(b, 2, 2);
assert(isequal(r2, [false false; false false]), 'repmat(false, 2, 2) should be 2x2 false matrix');
assert(islogical(r2), 'result should be logical');

% Test 3: repmat logical expression (as used in trigtech/mtimes)
x = repmat(true & false, 1, 3);
assert(isequal(x, [false false false]), 'repmat of logical expr should work');
assert(islogical(x), 'result should be logical');

% Test 4: repmat logical array
v = [true false true];
r3 = repmat(v, 2, 1);
assert(isequal(r3, [true false true; true false true]), 'repmat of logical row vector');

disp('SUCCESS')
