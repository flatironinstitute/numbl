% Test real SVD (works with pure TypeScript fallback)

% Test 1: singular values only for 2x2 matrix
A = [4, 3; 2, 1];
s = svd(A);
assert(length(s) == 2);
assert(s(1) >= s(2));
assert(s(2) > 0);
% Known singular values for [4,3;2,1]: sqrt(15+sqrt(200)) and sqrt(15-sqrt(200))
assert(abs(s(1) - 5.4649857042) < 1e-6);
assert(abs(s(2) - 0.3659662171) < 1e-6);

% Test 2: full SVD [U, S, V] = svd(A) for square matrix
A = [4, 3; 2, 1];
[U, S, V] = svd(A);
% Verify reconstruction: A = U * S * V'
assert(norm(U * S * V' - A) < 1e-10);
% U should be orthogonal
assert(norm(U' * U - eye(2)) < 1e-10);
% V should be orthogonal
assert(norm(V' * V - eye(2)) < 1e-10);
% S should be diagonal with non-negative entries
assert(S(1,1) >= S(2,2));
assert(S(2,2) >= 0);
assert(S(1,2) == 0);
assert(S(2,1) == 0);

% Test 3: full SVD for tall matrix (m > n)
T = [1, 2; 3, 4; 5, 6];
[U, S, V] = svd(T);
assert(size(U, 1) == 3);
assert(size(U, 2) == 3);
assert(size(S, 1) == 3);
assert(size(S, 2) == 2);
assert(size(V, 1) == 2);
assert(size(V, 2) == 2);
assert(norm(U * S * V' - T) < 1e-10);
assert(norm(U' * U - eye(3)) < 1e-10);
assert(norm(V' * V - eye(2)) < 1e-10);

% Test 4: full SVD for wide matrix (m < n)
W = [1, 2, 3; 4, 5, 6];
[U, S, V] = svd(W);
assert(size(U, 1) == 2);
assert(size(U, 2) == 2);
assert(size(S, 1) == 2);
assert(size(S, 2) == 3);
assert(size(V, 1) == 3);
assert(size(V, 2) == 3);
assert(norm(U * S * V' - W) < 1e-10);
assert(norm(U' * U - eye(2)) < 1e-10);
assert(norm(V' * V - eye(3)) < 1e-10);

% Test 5: economy SVD for tall matrix with 'econ'
T = [1, 2; 3, 4; 5, 6];
[U, S, V] = svd(T, 'econ');
assert(size(U, 1) == 3);
assert(size(U, 2) == 2);
assert(size(S, 1) == 2);
assert(size(S, 2) == 2);
assert(size(V, 1) == 2);
assert(size(V, 2) == 2);
assert(norm(U * S * V' - T) < 1e-10);

% Test 6: economy SVD with numeric 0 flag
T = [1, 2; 3, 4; 5, 6];
[U2, S2, V2] = svd(T, 0);
assert(size(U2, 1) == 3);
assert(size(U2, 2) == 2);
assert(norm(U2 * S2 * V2' - T) < 1e-10);

% Test 7: economy SVD for wide matrix
W = [1, 2, 3; 4, 5, 6];
[U, S, V] = svd(W, 'econ');
assert(size(U, 1) == 2);
assert(size(U, 2) == 2);
assert(size(S, 1) == 2);
assert(size(S, 2) == 2);
assert(size(V, 1) == 3);
assert(size(V, 2) == 2);
assert(norm(U * S * V' - W) < 1e-10);

% Test 8: identity matrix
I3 = eye(3);
s = svd(I3);
assert(norm(s - [1; 1; 1]) < 1e-10);
[U, S, V] = svd(I3);
assert(norm(U * S * V' - I3) < 1e-10);

% Test 9: singular matrix (rank deficient)
A = [1, 2; 2, 4];
s = svd(A);
assert(s(1) > 0);
assert(abs(s(2)) < 1e-10);
[U, S, V] = svd(A);
assert(norm(U * S * V' - A) < 1e-10);

% Test 10: 1x1 matrix
A = [5];
s = svd(A);
assert(abs(s - 5) < 1e-10);

% Test 11: larger matrix (5x5)
A = [16, 2, 3, 13, 5; 5, 11, 10, 8, 4; 9, 7, 6, 12, 3; 4, 14, 15, 1, 2; 1, 2, 3, 4, 5];
[U, S, V] = svd(A);
assert(size(U, 1) == 5);
assert(size(U, 2) == 5);
assert(norm(U * S * V' - A) < 1e-10);
assert(norm(U' * U - eye(5)) < 1e-10);
assert(norm(V' * V - eye(5)) < 1e-10);
% Singular values should be in descending order
for i = 1:4
    assert(S(i,i) >= S(i+1,i+1));
end

% Test 12: matrix with negative entries
A = [-3, 2; 1, -4];
[U, S, V] = svd(A);
assert(norm(U * S * V' - A) < 1e-10);
assert(S(1,1) >= 0);
assert(S(2,2) >= 0);

% Test 13: singular values via norm(A, 2) should equal largest singular value
A = [1, 2; 3, 4; 5, 6];
n2 = norm(A, 2);
s = svd(A);
assert(abs(n2 - s(1)) < 1e-10);

% Test 14: very tall matrix (10x2)
A = reshape(1:20, 10, 2);
[U, S, V] = svd(A, 'econ');
assert(size(U, 1) == 10);
assert(size(U, 2) == 2);
assert(norm(U * S * V' - A) < 1e-10);

% Test 15: very wide matrix (2x10)
A = reshape(1:20, 2, 10);
[U, S, V] = svd(A, 'econ');
assert(size(V, 1) == 10);
assert(size(V, 2) == 2);
assert(norm(U * S * V' - A) < 1e-10);

disp('SUCCESS');
