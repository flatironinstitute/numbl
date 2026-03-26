% Test 3-output column-pivoted QR: [Q, R, E] = qr(A, ...)

% --- Real matrices, economy mode (E is a permutation vector) ---

% Test 1: square real matrix, economy mode
A = [1 2 3; 4 5 6; 7 8 9];
[Q, R, E] = qr(A, 0);
assert(all(size(Q) == [3 3]), 'Q size wrong');
assert(all(size(R) == [3 3]), 'R size wrong');
assert(all(size(E) == [1 3]), 'E should be 1xn vector in econ mode');
% E is a permutation vector: A(:,E) = Q*R
assert(norm(A(:, E) - Q * R) < 1e-10, 'A(:,E) should equal Q*R (real econ)');
assert(norm(Q' * Q - eye(3)) < 1e-10, 'Q should be orthogonal (real econ)');

% Test 2: tall real matrix, economy mode
B = [1 2; 3 4; 5 6; 7 8];
[Q2, R2, E2] = qr(B, 0);
assert(all(size(Q2) == [4 2]), 'Q size wrong for tall');
assert(all(size(R2) == [2 2]), 'R size wrong for tall');
assert(all(size(E2) == [1 2]), 'E size wrong for tall');
assert(norm(B(:, E2) - Q2 * R2) < 1e-10, 'B(:,E) should equal Q*R (tall econ)');
assert(norm(Q2' * Q2 - eye(2)) < 1e-10, 'Q orthogonal (tall econ)');

% Test 3: full QR (non-economy), E is a permutation matrix
C = [3 1; 1 4; 0 2];
[Q3, R3, E3] = qr(C);
assert(all(size(Q3) == [3 3]), 'Q full size wrong');
assert(all(size(R3) == [3 2]), 'R full size wrong');
assert(all(size(E3) == [2 2]), 'E should be nxn matrix in full mode');
% A*E = Q*R for permutation matrix
assert(norm(C * E3 - Q3 * R3) < 1e-10, 'C*E should equal Q*R (full)');
assert(norm(Q3' * Q3 - eye(3)) < 1e-10, 'Q orthogonal (full)');
% E should be a valid permutation matrix
assert(norm(E3' * E3 - eye(2)) < 1e-10, 'E should be orthogonal');

% Test 4: permutation vector contains a valid permutation of 1:n
n = 5;
D = randn(8, n);
[Q4, R4, E4] = qr(D, 0);
assert(all(size(E4) == [1 n]), 'E4 size');
assert(length(unique(E4)) == n, 'E4 should be a permutation');
assert(min(E4) == 1 && max(E4) == n, 'E4 range should be 1:n');
assert(norm(D(:, E4) - Q4 * R4) < 1e-10, 'D(:,E) = Q*R (random)');

% Test 5: single-column matrix
v = [3; 1; 4; 1; 5];
[Q5, R5, E5] = qr(v, 0);
assert(E5 == 1, 'E should be [1] for single column');
assert(abs(norm(v) - abs(R5)) < 1e-10, 'R should be norm(v) for single column');

% Test 6: R upper triangular
F = randn(6, 4);
[Q6, R6, E6] = qr(F, 0);
for i = 2:4
    for j = 1:i-1
        assert(abs(R6(i, j)) < 1e-10, 'R should be upper triangular');
    end
end

% Test 7: full QR permutation matrix for square matrix
G = [1 3 2; 4 6 5; 7 9 8];
[Q7, R7, E7] = qr(G);
assert(all(size(E7) == [3 3]), 'E should be 3x3 for square full QR');
assert(norm(G * E7 - Q7 * R7) < 1e-10, 'G*E = Q*R (square full)');

% --- Complex matrices ---

% Skip complex tests if LAPACK addon not available
try
    qr([1+1i 0; 0 1], 0);
catch e
    if ~isempty(strfind(e.message, 'requires'))
        disp('SUCCESS');
        return;
    end
    rethrow(e);
end

% Test 8: complex square matrix, economy mode
H = [1+2i 3+4i; 5+6i 7+8i];
[Q8, R8, E8] = qr(H, 0);
assert(all(size(E8) == [1 2]), 'complex E size (econ)');
assert(norm(H(:, E8) - Q8 * R8) < 1e-10, 'complex A(:,E) = Q*R (econ)');
assert(norm(Q8' * Q8 - eye(2)) < 1e-10, 'complex Q unitary (econ)');

% Test 9: complex tall matrix
J = [1+1i 2-1i 3; 4 5+2i 6-1i; 7+3i 8 9+1i; 10-1i 11+1i 12];
[Q9, R9, E9] = qr(J, 0);
assert(all(size(Q9) == [4 3]), 'complex Q size');
assert(all(size(R9) == [3 3]), 'complex R size');
assert(norm(J(:, E9) - Q9 * R9) < 1e-10, 'complex J(:,E) = Q*R (tall)');
assert(norm(Q9' * Q9 - eye(3)) < 1e-10, 'complex Q unitary (tall)');

% Test 10: complex full QR returns permutation matrix
K = [1+1i 2; 3 4+1i; 5-1i 6];
[Q10, R10, E10] = qr(K);
assert(all(size(Q10) == [3 3]), 'complex full Q size');
assert(all(size(R10) == [3 2]), 'complex full R size');
assert(all(size(E10) == [2 2]), 'complex full E should be matrix');
assert(norm(K * E10 - Q10 * R10) < 1e-10, 'complex K*E = Q*R (full)');
assert(norm(Q10' * Q10 - eye(3)) < 1e-10, 'complex Q unitary (full)');

% Test 11: purely imaginary matrix
L = [1i 2i 3i; 4i 5i 6i];
[Q11, R11, E11] = qr(L, 0);
assert(norm(L(:, E11) - Q11 * R11) < 1e-10, 'imaginary A(:,E) = Q*R');

disp('SUCCESS');
