% Test lu factorization

% Test 1: [L, U, P] = lu(A) for a 3x3 matrix
A = [2 1 1; 4 3 3; 8 7 9];
[L, U, P] = lu(A);
assert(norm(P * A - L * U) < 1e-10, 'P*A should equal L*U');
% L should be unit lower triangular
for i = 1:size(L, 1)
    assert(abs(L(i, i) - 1) < 1e-10, 'L diagonal should be 1');
end
% Check L is lower triangular
for i = 1:size(L, 1)
    for j = i+1:size(L, 2)
        assert(abs(L(i, j)) < 1e-10, 'L should be lower triangular');
    end
end
% Check U is upper triangular
for i = 2:size(U, 1)
    for j = 1:i-1
        assert(abs(U(i, j)) < 1e-10, 'U should be upper triangular');
    end
end

% Test 2: [L, U] = lu(A) — two-output form
[L2, U2] = lu(A);
assert(norm(A - L2 * U2) < 1e-10, 'A should equal L*U (two-output form)');

% Test 3: [L, U, P] = lu(A, 'vector') — permutation vector form
[L3, U3, p] = lu(A, 'vector');
assert(norm(A(p, :) - L3 * U3) < 1e-10, 'A(p,:) should equal L*U (vector form)');

% Test 4: Scalar
[L4, U4, P4] = lu(5);
assert(L4 == 1, 'Scalar L should be 1');
assert(U4 == 5, 'Scalar U should be 5');
assert(P4 == 1, 'Scalar P should be 1');

% Test 5: Identity matrix
I3 = eye(3);
[L5, U5, P5] = lu(I3);
assert(norm(P5 * I3 - L5 * U5) < 1e-10, 'P*I should equal L*U');

% Test 6: Non-square matrix (tall)
B = [1 2; 3 4; 5 6];
[L6, U6, P6] = lu(B);
assert(norm(P6 * B - L6 * U6) < 1e-10, 'P*B should equal L*U for tall matrix');

% Test 7: Non-square matrix (wide)
C = [1 2 3; 4 5 6];
[L7, U7, P7] = lu(C);
assert(norm(P7 * C - L7 * U7) < 1e-10, 'P*C should equal L*U for wide matrix');

% Test 8: 2x2 matrix
D = [0 1; 1 0];
[L8, U8, P8] = lu(D);
assert(norm(P8 * D - L8 * U8) < 1e-10, 'P*D should equal L*U for 2x2');

fprintf('SUCCESS\n');
