% Test pagetranspose

% Test 1: 2D matrix (single page)
A = [1 2; 3 4];
B = pagetranspose(A);
assert(isequal(B, [1 3; 2 4]));

% Test 2: 3D array (multiple pages)
X = zeros(2, 3, 2);
X(:,:,1) = [1 2 3; 4 5 6];
X(:,:,2) = [7 8 9; 10 11 12];
Y = pagetranspose(X);
assert(isequal(size(Y), [3, 2, 2]));
assert(isequal(Y(:,:,1), [1 4; 2 5; 3 6]));
assert(isequal(Y(:,:,2), [7 10; 8 11; 9 12]));

% Test 3: Vector (1xN)
v = [1 2 3];
assert(isequal(pagetranspose(v), [1; 2; 3]));

fprintf('SUCCESS\n');
