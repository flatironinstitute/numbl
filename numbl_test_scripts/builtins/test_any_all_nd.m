% Test any/all on N-D arrays with shape(1)==1
% MATLAB reduces along first non-singleton dimension for non-vector inputs

% 3D array with shape [1, 3, 2] -- NOT a vector
A = reshape([1 0 1 0 1 0], [1, 3, 2]);
assert(isequal(size(A), [1, 3, 2]));

% any(A) should reduce along dim 2 (first non-singleton), giving [1, 1, 2]
r = any(A);
assert(isequal(size(r), [1, 1, 2]), 'any: wrong output shape for 3D array');

% all(A) should also reduce along dim 2
r2 = all(A);
assert(isequal(size(r2), [1, 1, 2]), 'all: wrong output shape for 3D array');

% 2D row vector [1, 4] -- IS a vector, should return scalar
B = [1 0 1 1];
assert(isequal(size(B), [1, 4]));
r3 = any(B);
assert(isscalar(r3), 'any: row vector should return scalar');
assert(r3 == true);

r4 = all(B);
assert(isscalar(r4), 'all: row vector should return scalar');
assert(r4 == false);

% Column vector [3, 1] -- should return scalar
C = [1; 0; 1];
r5 = any(C);
assert(isscalar(r5));
assert(r5 == true);

% 3D array [1, 2, 3] with all ones
D = ones(1, 2, 3);
r6 = all(D);
assert(isequal(size(r6), [1, 1, 3]), 'all: wrong shape for 3D ones');
assert(all(r6(:)));

% 3D array [1, 2, 3] with all zeros
E = zeros(1, 2, 3);
r7 = any(E);
assert(isequal(size(r7), [1, 1, 3]), 'any: wrong shape for 3D zeros');
assert(~any(r7(:)));

disp('SUCCESS');
