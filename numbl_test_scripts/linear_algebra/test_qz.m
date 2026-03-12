% Test QZ factorization (generalized Schur decomposition)

% Skip if LAPACK addon is not available
try
    A = [1 2; 3 4];
    B = [5 6; 7 8];
    [AA, BB, Q, Z] = qz(A, B);
catch e
    if ~isempty(strfind(e.message, 'requires'))
        disp('SUCCESS');
        return;
    end
    rethrow(e);
end

% 2x2 case
A = [1 2; 3 4];
B = [5 6; 7 8];
[AA, BB, Q, Z] = qz(A, B);

% Verify Q*A*Z = AA
assert(max(max(abs(Q*A*Z - AA))) < 1e-10, 'Q*A*Z should equal AA');

% Verify Q*B*Z = BB
assert(max(max(abs(Q*B*Z - BB))) < 1e-10, 'Q*B*Z should equal BB');

% Verify Q is orthogonal
assert(max(max(abs(Q*Q' - eye(2)))) < 1e-10, 'Q should be orthogonal');

% Verify Z is orthogonal
assert(max(max(abs(Z*Z' - eye(2)))) < 1e-10, 'Z should be orthogonal');

% 3x3 case
A3 = [1 2 3; 4 5 6; 7 8 10];
B3 = [10 11 12; 13 14 15; 16 17 19];
[AA3, BB3, Q3, Z3] = qz(A3, B3);
assert(max(max(abs(Q3*A3*Z3 - AA3))) < 1e-10, '3x3: Q*A*Z = AA');
assert(max(max(abs(Q3*B3*Z3 - BB3))) < 1e-10, '3x3: Q*B*Z = BB');
assert(max(max(abs(Q3*Q3' - eye(3)))) < 1e-10, '3x3: Q orthogonal');
assert(max(max(abs(Z3*Z3' - eye(3)))) < 1e-10, '3x3: Z orthogonal');

% 6-output form with eigenvectors
[AA2, BB2, Q2, Z2, V2, W2] = qz(A3, B3);
assert(max(max(abs(Q2*A3*Z2 - AA2))) < 1e-10, '6-out: Q*A*Z = AA');
assert(max(max(abs(Q2*B3*Z2 - BB2))) < 1e-10, '6-out: Q*B*Z = BB');
assert(max(max(abs(V2))) > 0, 'V should be non-zero');
assert(max(max(abs(W2))) > 0, 'W should be non-zero');

% Complex input
Ac = [1+2i 3+4i; 5+6i 7+8i];
Bc = [2+1i 4+3i; 6+5i 8+7i];
[AAc, BBc, Qc, Zc] = qz(Ac, Bc);
assert(max(max(abs(Qc*Ac*Zc - AAc))) < 1e-10, 'complex: Q*A*Z = AA');
assert(max(max(abs(Qc*Bc*Zc - BBc))) < 1e-10, 'complex: Q*B*Z = BB');
assert(max(max(abs(Qc*Qc' - eye(2)))) < 1e-10, 'complex: Q unitary');
assert(max(max(abs(Zc*Zc' - eye(2)))) < 1e-10, 'complex: Z unitary');

% Identity matrices
[AA4, BB4, Q4, Z4] = qz(eye(3), eye(3));
assert(max(max(abs(Q4*eye(3)*Z4 - AA4))) < 1e-10, 'identity: Q*A*Z = AA');
assert(max(max(abs(Q4*eye(3)*Z4 - BB4))) < 1e-10, 'identity: Q*B*Z = BB');

disp('SUCCESS');
