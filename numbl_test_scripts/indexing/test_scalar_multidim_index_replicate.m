% A scalar (1x1) indexed with MULTIPLE subscripts must replicate, just like
% the single-subscript case.  Every numeric value is a 1x1 matrix, so for a
% scalar r:
%   r([1;1;1], 1)   -> 3x1 column of r
%   r(1, [1 1 1])   -> 1x3 row of r
%   r([1;1;1], :)   -> 3x1 (colon over the single column => length 1)
% The result shape is [len(I), len(J)] with a colon/scalar subscript counting
% as length 1, filled with the scalar value.
%
% This is the mesh2d/aabb-tree findball.m "ballkern" pattern:
%   pk = pk.';  pk = pk(ones(mb,1), :);
% which hit a scalar pk and threw "Cannot convert non-scalar tensor to number".

r = 5;

% --- row replicate via column index, colon over the single column ---
a = r([1; 1; 1], 1);
assert(isequal(size(a), [3 1]), 'a shape');
assert(isequal(a, [5; 5; 5]), 'a values');

b = r(1, [1 1 1]);
assert(isequal(size(b), [1 3]), 'b shape');
assert(isequal(b, [5 5 5]), 'b values');

c = r([1 1], [1 1 1]);
assert(isequal(size(c), [2 3]), 'c shape');
assert(all(c(:) == 5), 'c values');

d = r([1; 1; 1], :);
assert(isequal(size(d), [3 1]), 'd shape');
assert(isequal(d, [5; 5; 5]), 'd values');

e = r(:, [1 1]);
assert(isequal(size(e), [1 2]), 'e shape');
assert(isequal(e, [5 5]), 'e values');

% --- trivial scalar subscripting still returns the scalar ---
f = r(1, 1);
assert(isequal(f, 5), 'f scalar');
g = r(:, :);
assert(isequal(g, 5), 'g scalar');

% --- complex scalar replicates real and imaginary parts ---
z = 3 + 4i;
h = z([1; 1; 1], 1);
assert(isequal(size(h), [3 1]), 'h shape');
assert(isequal(h, [3+4i; 3+4i; 3+4i]), 'h complex values');

% --- scalar logical base replicates and stays logical ---
tb = true;
p = tb([1; 1; 1], 1);
assert(islogical(p), 'p logical');
assert(isequal(size(p), [3 1]), 'p shape');
assert(all(p), 'p values');
q = tb(1, [1 1]);
assert(islogical(q), 'q logical');
assert(isequal(size(q), [1 2]), 'q shape');
sgl = tb([1; 1; 1]);          % single index follows index shape
assert(islogical(sgl) && isequal(size(sgl), [3 1]), 'logical single-index shape');

% --- the exact mesh2d ballkern pattern ---
pk = 7;            % scalar after a tile collapses to one point
mb = 4;
pk = pk.';
pk = pk(ones(mb, 1), :);
assert(isequal(size(pk), [4 1]), 'ballkern shape');
assert(all(pk == 7), 'ballkern values');

% --- out-of-bounds in a subscript must still error ---
caught = false;
try
    bad = r([1 2], 1);
    disp(bad);
catch
    caught = true;
end
assert(caught, 'out-of-bounds subscript should error');

disp('SUCCESS');
