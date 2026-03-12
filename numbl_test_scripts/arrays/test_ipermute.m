% Test ipermute builtin

% 2D: ipermute is inverse of permute
A = [1 2 3; 4 5 6];
B = permute(A, [2 1]);
C = ipermute(B, [2 1]);
assert(isequal(C, A));

% 3D array
A3 = reshape(1:24, [2, 3, 4]);
dimorder = [3 1 2];
B3 = permute(A3, dimorder);
C3 = ipermute(B3, dimorder);
assert(isequal(C3, A3));

% Verify sizes
assert(isequal(size(B3), [4, 2, 3]));
assert(isequal(size(C3), [2, 3, 4]));

% Scalar
s = ipermute(5, [1 2]);
assert(s == 5);

fprintf('SUCCESS\n');
