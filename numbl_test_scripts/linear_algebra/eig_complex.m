% Test complex eig decomposition (requires LAPACK addon)

% Skip if LAPACK addon is not available
try
    eig([1+1i, 0; 0, 1+1i]);
catch e
    if ~isempty(strfind(e.message, 'requires'))
        disp('SUCCESS');
        return;
    end
    rethrow(e);
end

% Test 1: eigenvalues of a 2x2 Hermitian matrix
A = [2, 1+1i; 1-1i, 3];
e = eig(A);
e_sorted = sort(real(e));
% Eigenvalues of Hermitian matrix are real
assert(norm(imag(e)) < 1e-10, 'Hermitian eigenvalues should be real');
% trace = sum of eigenvalues = 5, det = 6 - 2 = 4
assert(abs(sum(e_sorted) - 5) < 1e-10, 'Sum of eigenvalues should equal trace');
assert(abs(prod(e_sorted) - 4) < 1e-10, 'Product of eigenvalues should equal det');

% Test 2: [V, D] = eig(A) for Hermitian matrix — V*D*V' should recover A
[V, D] = eig(A);
assert(norm(V * D * inv(V) - A) < 1e-10, 'V*D*inv(V) should equal A for Hermitian');

% Test 3: purely imaginary matrix [[0, -i], [i, 0]] — eigenvalues ±1
B = [0, -1i; 1i, 0];
e2 = eig(B);
e2_sorted = sort(real(e2));
assert(abs(e2_sorted(1) - (-1)) < 1e-10, 'First eigenvalue should be -1');
assert(abs(e2_sorted(2) - 1) < 1e-10, 'Second eigenvalue should be 1');

% Test 4: [V, D] = eig for non-Hermitian complex matrix
C = [1+2i, 3+4i; 5+6i, 7+8i];
[V3, D3] = eig(C);
assert(norm(C * V3 - V3 * D3) < 1e-8, 'A*V should equal V*D for non-Hermitian complex');

% Test 5: eigenvalues only for 3x3 complex matrix
E = [2+1i, 1+3i, 0; 1, 3+2i, 1+1i; 0, 1+1i, 4+1i];
e4 = eig(E);
% Eigenvalues should sum to trace
tr = E(1,1) + E(2,2) + E(3,3);
assert(abs(sum(e4) - tr) < 1e-10, 'Sum of eigenvalues should equal trace for 3x3');

% Test 6: [V, D] for 3x3 complex matrix
[V5, D5] = eig(E);
assert(norm(E * V5 - V5 * D5) < 1e-8, 'A*V should equal V*D for 3x3 complex');

% Test 7: 'vector' output form
[V6, d6] = eig(C, 'vector');
assert(size(d6, 2) == 1, 'vector form should return column vector');
assert(size(d6, 1) == 2, 'vector form should have n elements');

% Test 8: 3-output form [V, D, W]
[V7, D7, W7] = eig(C);
assert(norm(C * V7 - V7 * D7) < 1e-8, 'A*V should equal V*D for 3-output form');
% W are left eigenvectors: W'*A = D*W'
assert(norm(W7' * C - D7 * W7') < 1e-8, 'W''*A should equal D*W'' for left eigenvectors');

% Test 9: complex identity matrix
I3 = eye(3) + 0i * eye(3);
e5 = eig(I3);
assert(norm(e5 - ones(3, 1)) < 1e-10, 'Eigenvalues of complex identity should be 1');

% Test 10: diagonal complex matrix — eigenvalues are the diagonal entries
F = diag([1+2i, 3+4i, 5+6i]);
e6 = eig(F);
e6_sorted = sort(real(e6));
expected = sort([1; 3; 5]);
assert(norm(real(sort(e6)) - expected) < 1e-10, 'Eigenvalues of diag matrix should match diagonal (real)');
assert(norm(sort(imag(e6)) - sort([2; 4; 6])) < 1e-10, 'Eigenvalues of diag matrix should match diagonal (imag)');

disp('SUCCESS');
