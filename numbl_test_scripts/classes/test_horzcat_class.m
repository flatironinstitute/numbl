% Test that [a, b] on class instances dispatches to horzcat method.

% Test 1: [a, b] should call horzcat and combine data
a = HorzCat_([1, 2]);
b = HorzCat_([3, 4]);
c = [a, b];
assert(isequal(c.data, [1, 2, 3, 4]), 'horzcat should combine data arrays');

% Test 2: [a, b, c_obj] with three instances
d = HorzCat_([5]);
e = [a, b, d];
assert(isequal(e.data, [1, 2, 3, 4, 5]), 'horzcat should work with three operands');

% Test 3: Single element — [a] should just return a
f = [a];
assert(isequal(f.data, [1, 2]), 'single-element horzcat should return original');

disp('SUCCESS')
