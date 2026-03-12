% Test complex QR decomposition (requires LAPACK addon)

% Skip if LAPACK addon is not available
try
    qr([1+1i, 0; 0, 1+1i]);
catch e
    if ~isempty(strfind(e.message, 'requires'))
        disp('SUCCESS');
        return;
    end
    rethrow(e);
end

% Test 1: full QR of complex 2x2 matrix
A = [1+2i, 3+4i; 5+6i, 7+8i];
[Q, R] = qr(A);
% Q*R should recover A
assert(norm(Q * R - A) < 1e-10, 'Q*R should equal A for 2x2 complex');
% Q should be unitary: Q'*Q = I
assert(norm(Q' * Q - eye(2)) < 1e-10, 'Q should be unitary for 2x2 complex');
% R should be upper triangular
assert(abs(R(2, 1)) < 1e-10, 'R should be upper triangular');

% Test 2: full QR of complex 3x3 matrix
B = [2+1i, 1+3i, 0; 1, 3+2i, 1+1i; 0, 1+1i, 4+1i];
[Q2, R2] = qr(B);
assert(norm(Q2 * R2 - B) < 1e-10, 'Q*R should equal B for 3x3 complex');
assert(norm(Q2' * Q2 - eye(3)) < 1e-10, 'Q should be unitary for 3x3 complex');
% Check R is upper triangular
for i = 2:3
    for j = 1:i-1
        assert(abs(R2(i, j)) < 1e-10, 'R should be upper triangular');
    end
end

% Test 3: economy QR of tall complex matrix (4x2)
C = [1+1i, 2+2i; 3+3i, 4+4i; 5+5i, 6+6i; 7+7i, 8+8i];
[Q3, R3] = qr(C, 0);
% Dimensions: Q is 4x2, R is 2x2
assert(size(Q3, 1) == 4, 'Q rows should be m');
assert(size(Q3, 2) == 2, 'Q cols should be k');
assert(size(R3, 1) == 2, 'R rows should be k');
assert(size(R3, 2) == 2, 'R cols should be n');
% Verify reconstruction
assert(norm(Q3 * R3 - C) < 1e-10, 'Q*R should equal C for economy QR');
% Q columns should be orthonormal
assert(norm(Q3' * Q3 - eye(2)) < 1e-10, 'Q columns should be orthonormal');

% Test 4: economy QR with 'econ' string
[Q4, R4] = qr(C, 'econ');
assert(norm(Q4 * R4 - C) < 1e-10, 'Q*R should equal C for econ QR');

% Test 5: wide complex matrix (2x4)
D = [1+1i, 2-1i, 3+2i, 4; 5, 6+3i, 7-1i, 8+2i];
[Q5, R5] = qr(D);
assert(norm(Q5 * R5 - D) < 1e-10, 'Q*R should equal D for wide complex');
assert(norm(Q5' * Q5 - eye(2)) < 1e-10, 'Q should be unitary for wide complex');

% Test 6: single output (R only)
R6 = qr(A);
assert(size(R6, 1) == 2, 'R rows should be m for single output');
assert(size(R6, 2) == 2, 'R cols should be n for single output');

% Test 7: identity complex matrix
I2 = eye(2) + 0i * eye(2);
[Q7, R7] = qr(I2);
% Q and R should each be close to identity (up to sign)
assert(norm(Q7 * R7 - I2) < 1e-10, 'Q*R should equal I for complex identity');

% Test 8: purely imaginary matrix
E = [1i, 2i; 3i, 4i];
[Q8, R8] = qr(E);
assert(norm(Q8 * R8 - E) < 1e-10, 'Q*R should equal E for purely imaginary');
assert(norm(Q8' * Q8 - eye(2)) < 1e-10, 'Q should be unitary for purely imaginary');

% Test 9: diagonal dominant complex matrix (well-conditioned)
n = 5;
F = rand(n) + 1i * rand(n) + n * eye(n);
[Q9, R9] = qr(F);
assert(norm(Q9 * R9 - F) < 1e-10, 'Q*R should equal F for 5x5 complex');
assert(norm(Q9' * Q9 - eye(n)) < 1e-10, 'Q should be unitary for 5x5 complex');

disp('SUCCESS');
