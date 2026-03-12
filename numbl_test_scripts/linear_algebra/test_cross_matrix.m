% Test cross product: shape preservation and matrix column-wise operation

% cross on row vectors (basic)
r1 = cross([1 0 0], [0 1 0]);
assert(isequal(r1, [0 0 1]), 'cross i x j = k');

r2 = cross([1 2 3], [4 5 6]);
assert(isequal(r2, [2*6-3*5, 3*4-1*6, 1*5-2*4]), 'cross product formula');

% cross on column vectors should return column vector
r3 = cross([1; 0; 0], [0; 1; 0]);
assert(isequal(size(r3), [3 1]), 'cross col vectors -> col result');
assert(isequal(r3, [0; 0; 1]), 'cross col vectors value');

r4 = cross([1; 2; 3], [4; 5; 6]);
assert(isequal(size(r4), [3 1]), 'cross col result shape');
assert(isequal(r4, [2*6-3*5; 3*4-1*6; 1*5-2*4]), 'cross col result values');

% cross on 3xN matrices (column-wise cross products)
A = [1 0; 0 1; 0 0];
B = [0 0; 1 0; 0 1];
r5 = cross(A, B);
assert(isequal(size(r5), [3 2]), 'cross matrix size');
assert(isequal(r5(:,1), [0; 0; 1]), 'cross matrix col1');
assert(isequal(r5(:,2), [1; 0; 0]), 'cross matrix col2');

disp('SUCCESS');
