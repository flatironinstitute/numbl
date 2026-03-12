% Test that [a; b] on class instances dispatches to vertcat method.

% Test 1: [a; b] should call vertcat and combine data
a = VertCat_([1, 2]);
b = VertCat_([3, 4]);
c = [a; b];
assert(isequal(c.data, [1, 2; 3, 4]), 'vertcat should stack data arrays as rows');

% Test 2: [a; b; c_obj] with three instances
d = VertCat_([5, 6]);
e = [a; b; d];
assert(isequal(e.data, [1, 2; 3, 4; 5, 6]), 'vertcat should work with three operands');

% Test 3: Single element — [a] should just return a (no vertcat call needed)
f = [a];
assert(isequal(f.data, [1, 2]), 'single-element should return original');

disp('SUCCESS')
