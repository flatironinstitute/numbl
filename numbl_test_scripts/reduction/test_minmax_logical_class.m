% min/max of logical inputs should return a logical, not a double.
% MATLAB: class(min(true, false)) == 'logical'.

% ── Two-argument min/max on logical scalars ────────────────────────────
a = min(true, false);
assert(islogical(a), 'min(true,false) should be logical');
assert(a == false, 'min(true,false) should equal false');

b = max(true, false);
assert(islogical(b), 'max(true,false) should be logical');
assert(b == true, 'max(true,false) should equal true');

% Single arg on a logical row
c = min([true, false, true]);
assert(islogical(c), 'min of a logical row should be logical');
assert(c == false, 'min of a logical row should be false');

d = max([false, false, true]);
assert(islogical(d), 'max of a logical row should be logical');
assert(d == true, 'max of a logical row should be true');

% Two-argument min/max on logical tensors
e = min([true false true], [false true true]);
assert(islogical(e), 'element-wise min of logical vectors should be logical');
assert(isequal(e, [false false true]), 'element-wise min values wrong');

f = max([true false true], [false true true]);
assert(islogical(f), 'element-wise max of logical vectors should be logical');
assert(isequal(f, [true true true]), 'element-wise max values wrong');

% Mixed: logical and logical (broadcast with scalar)
g = min([true, true, false], false);
assert(islogical(g), 'min(logical row, logical scalar) should be logical');
assert(isequal(g, [false false false]), 'min broadcast values wrong');

% Mixed logical + numeric: MATLAB promotes to double
h = min(true, 0.5);
assert(strcmp(class(h), 'double'), 'min(true, 0.5) should be double');
assert(h == 0.5, 'min(true, 0.5) should equal 0.5');

disp('SUCCESS')
