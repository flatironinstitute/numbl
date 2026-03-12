% Test complex LU factorization (requires LAPACK addon)

% Skip if LAPACK addon is not available
try
    [L, U, P] = lu([1+1i, 0; 0, 1+1i]);
catch e
    if ~isempty(strfind(e.message, 'requires'))
        disp('SUCCESS');
        return;
    end
    rethrow(e);
end

% Test 1: [L, U, P] = lu(A) for 3x3 complex matrix
A = [2+1i, 1+3i, 0; 1, 3+2i, 1+1i; 0, 1+1i, 4+1i];
[L, U, P] = lu(A);
assert(norm(P * A - L * U) < 1e-10, 'P*A should equal L*U for 3x3 complex');
% L should be unit lower triangular
for i = 1:size(L, 1)
    assert(abs(L(i, i) - 1) < 1e-10, 'L diagonal should be 1');
end
for i = 1:size(L, 1)
    for j = i+1:size(L, 2)
        assert(abs(L(i, j)) < 1e-10, 'L should be lower triangular');
    end
end
% U should be upper triangular
for i = 2:size(U, 1)
    for j = 1:i-1
        assert(abs(U(i, j)) < 1e-10, 'U should be upper triangular');
    end
end

% Test 2: [L, U] = lu(A) — two-output form
[L2, U2] = lu(A);
assert(norm(A - L2 * U2) < 1e-10, 'A should equal L*U (two-output complex)');

% Test 3: [L, U, P] = lu(A, 'vector') — permutation vector form
[L3, U3, p] = lu(A, 'vector');
assert(norm(A(p, :) - L3 * U3) < 1e-10, 'A(p,:) should equal L*U (vector form complex)');

% Test 4: purely imaginary matrix
B = [2i, 1i; 3i, 4i];
[L4, U4, P4] = lu(B);
assert(norm(P4 * B - L4 * U4) < 1e-10, 'P*B should equal L*U for purely imaginary');

% Test 5: complex identity matrix
I3 = eye(3) + 0i * eye(3);
[L5, U5, P5] = lu(I3);
assert(norm(P5 * I3 - L5 * U5) < 1e-10, 'P*I should equal L*U for complex identity');

% Test 6: 2x2 complex matrix
C = [1+2i, 3+4i; 5+6i, 7+8i];
[L6, U6, P6] = lu(C);
assert(norm(P6 * C - L6 * U6) < 1e-10, 'P*C should equal L*U for 2x2 complex');

% Test 7: tall complex matrix (4x2)
D = [1+1i, 2+2i; 3+3i, 4+4i; 5+5i, 6+6i; 7+7i, 8+8i];
[L7, U7, P7] = lu(D);
assert(norm(P7 * D - L7 * U7) < 1e-10, 'P*D should equal L*U for tall complex');
assert(size(L7, 1) == 4, 'L rows should be m for tall matrix');
assert(size(L7, 2) == 2, 'L cols should be k for tall matrix');
assert(size(U7, 1) == 2, 'U rows should be k for tall matrix');
assert(size(U7, 2) == 2, 'U cols should be n for tall matrix');

% Test 8: wide complex matrix (2x4)
E = [1+1i, 2-1i, 3+2i, 4; 5, 6+3i, 7-1i, 8+2i];
[L8, U8, P8] = lu(E);
assert(norm(P8 * E - L8 * U8) < 1e-10, 'P*E should equal L*U for wide complex');
assert(size(L8, 1) == 2, 'L rows should be m for wide matrix');
assert(size(L8, 2) == 2, 'L cols should be k for wide matrix');
assert(size(U8, 1) == 2, 'U rows should be k for wide matrix');
assert(size(U8, 2) == 4, 'U cols should be n for wide matrix');

% Test 9: diagonal dominant complex matrix (well-conditioned, larger)
n = 5;
F = rand(n) + 1i * rand(n) + n * eye(n);
[L9, U9, P9] = lu(F);
assert(norm(P9 * F - L9 * U9) < 1e-10, 'P*F should equal L*U for 5x5 complex');

% Test 10: two-output form for larger matrix
[L10, U10] = lu(F);
assert(norm(F - L10 * U10) < 1e-10, 'F should equal L*U (two-output 5x5 complex)');

disp('SUCCESS');
