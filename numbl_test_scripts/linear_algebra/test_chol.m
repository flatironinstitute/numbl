% Test chol (Cholesky factorization)

% Skip if LAPACK addon is not available
try
    chol([1 0; 0 1]);
catch e
    if ~isempty(strfind(e.message, 'requires')) || ~isempty(strfind(e.message, 'not available'))
        disp('SUCCESS');
        return;
    end
    rethrow(e);
end

% Test 1: R = chol(A) — upper triangular factor
A = [4 2; 2 3];
R = chol(A);
assert(norm(R' * R - A) < 1e-10, 'R''*R should equal A');
% R should be upper triangular
assert(abs(R(2, 1)) < 1e-10, 'R should be upper triangular');

% Test 2: R = chol(A, 'upper') — same as default
R2 = chol(A, 'upper');
assert(norm(R2 - R) < 1e-10, 'chol(A,''upper'') should equal chol(A)');

% Test 3: L = chol(A, 'lower') — lower triangular factor
L = chol(A, 'lower');
assert(norm(L * L' - A) < 1e-10, 'L*L'' should equal A');
% L should be lower triangular
assert(abs(L(1, 2)) < 1e-10, 'L should be lower triangular');
% L should equal R'
assert(norm(L - R') < 1e-10, 'L should equal R''');

% Test 4: Larger matrix (3x3)
B = [25 15 -5; 15 18 0; -5 0 11];
R3 = chol(B);
assert(norm(R3' * R3 - B) < 1e-10, 'R''*R should equal B for 3x3');
L3 = chol(B, 'lower');
assert(norm(L3 * L3' - B) < 1e-10, 'L*L'' should equal B for 3x3');

% Test 5: Identity matrix
I3 = eye(3);
R4 = chol(I3);
assert(norm(R4 - eye(3)) < 1e-10, 'chol(eye(3)) should be eye(3)');

% Test 6: Scalar
R5 = chol(9);
assert(abs(R5 - 3) < 1e-10, 'chol(9) should be 3');

% Test 7: [R, flag] = chol(A) — positive definite matrix (flag should be 0)
[R6, flag6] = chol(A);
assert(flag6 == 0, 'flag should be 0 for positive definite matrix');
assert(norm(R6' * R6 - A) < 1e-10, 'R''*R should equal A with flag output');

% Test 8: [R, flag] = chol(A) — non-positive-definite matrix (flag > 0)
C = [1 2; 2 1];
[R7, flag7] = chol(C);
assert(flag7 > 0, 'flag should be > 0 for non-positive-definite matrix');

% Test 9: [R, flag] = chol(A, 'lower') — with flag and lower
[L8, flag8] = chol(A, 'lower');
assert(flag8 == 0, 'flag should be 0 for positive definite matrix (lower)');
assert(norm(L8 * L8' - A) < 1e-10, 'L*L'' should equal A with flag output');

% Test 10: Complex Hermitian positive definite matrix
D = [2 1+1i; 1-1i 3];
R9 = chol(D);
assert(norm(R9' * R9 - D) < 1e-10, 'R''*R should equal D for complex matrix');

% Test 11: Complex lower Cholesky
L9 = chol(D, 'lower');
assert(norm(L9 * L9' - D) < 1e-10, 'L*L'' should equal D for complex matrix');
assert(norm(L9 - R9') < 1e-10, 'L should equal R'' for complex matrix');

% Test 12: Complex with flag output
[R10, flag10] = chol(D);
assert(flag10 == 0, 'flag should be 0 for complex positive definite');

% Test 13: 4x4 positive definite matrix
E = [10 3 1 0; 3 10 2 1; 1 2 10 4; 0 1 4 10];
R11 = chol(E);
assert(norm(R11' * R11 - E) < 1e-10, 'R''*R should equal E for 4x4');

% Test 14: 1x1 matrix
F = [4];
R12 = chol(F);
assert(abs(R12 - 2) < 1e-10, 'chol([4]) should be [2]');

% Test 15: Non-positive-definite should not error with two outputs
[R13, flag13] = chol(-eye(3));
assert(flag13 == 1, 'flag should be 1 for negative definite matrix');

fprintf('SUCCESS\n');
