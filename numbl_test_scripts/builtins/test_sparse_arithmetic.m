% Test sparse matrix arithmetic — Phase 2

%% Sparse + Sparse
A = sparse([1 2 3], [1 2 3], [10 20 30], 3, 3);
B = sparse([1 2 3], [1 2 3], [1 2 3], 3, 3);
C = A + B;
assert(issparse(C));
assert(isequal(full(C), [11 0 0; 0 22 0; 0 0 33]));

%% Sparse + Sparse with different patterns
A2 = sparse([1 3], [1 2], [5 7], 3, 3);
B2 = sparse([2 3], [1 3], [6 8], 3, 3);
C2 = A2 + B2;
assert(issparse(C2));
F2 = full(C2);
assert(F2(1,1) == 5);
assert(F2(2,1) == 6);
assert(F2(3,2) == 7);
assert(F2(3,3) == 8);
assert(nnz(C2) == 4);

%% Sparse + Sparse with cancellation
A3 = sparse([1 2], [1 1], [5 -3], 2, 2);
B3 = sparse([1 2], [1 1], [-5 3], 2, 2);
C3 = A3 + B3;
assert(issparse(C3));
assert(nnz(C3) == 0);
assert(isequal(full(C3), zeros(2, 2)));

%% Sparse - Sparse
D = A - B;
assert(issparse(D));
assert(isequal(full(D), [9 0 0; 0 18 0; 0 0 27]));

%% Sparse - Sparse with different patterns
D2 = A2 - B2;
assert(issparse(D2));
FD2 = full(D2);
assert(FD2(1,1) == 5);
assert(FD2(2,1) == -6);
assert(FD2(3,2) == 7);
assert(FD2(3,3) == -8);

%% Negation
N = -A;
assert(issparse(N));
assert(isequal(full(N), [-10 0 0; 0 -20 0; 0 0 -30]));

%% Sparse * scalar
S1 = A * 3;
assert(issparse(S1));
assert(isequal(full(S1), [30 0 0; 0 60 0; 0 0 90]));

%% scalar * Sparse
S2 = 3 * A;
assert(issparse(S2));
assert(isequal(full(S2), [30 0 0; 0 60 0; 0 0 90]));

%% Sparse * 0
S3 = A * 0;
assert(issparse(S3));
assert(nnz(S3) == 0);

%% Sparse * Sparse (matrix multiply)
P = sparse([1 1 2], [1 2 2], [1 2 3], 2, 2);
Q = sparse([1 2], [1 2], [4 5], 2, 2);
R = P * Q;
assert(issparse(R));
expected_R = full(P) * full(Q);
assert(isequal(full(R), expected_R));

%% Sparse * Sparse (non-square)
M1 = sparse([1 2], [1 3], [2 3], 2, 3);
M2 = sparse([1 3], [1 1], [4 5], 3, 1);
M3 = M1 * M2;
assert(issparse(M3));
assert(isequal(full(M3), full(M1) * full(M2)));

%% Sparse * Dense (matrix multiply)
SD = A * eye(3);
assert(~issparse(SD));
assert(isequal(SD, full(A)));

%% Dense * Sparse
DS = eye(3) * A;
assert(~issparse(DS));
assert(isequal(DS, full(A)));

%% Sparse + scalar → dense
R1 = A + 1;
assert(~issparse(R1));
assert(isequal(R1, full(A) + 1));

%% scalar + Sparse → dense
R2 = 1 + A;
assert(~issparse(R2));
assert(isequal(R2, 1 + full(A)));

%% Sparse - scalar → dense
R3 = A - 1;
assert(~issparse(R3));
assert(isequal(R3, full(A) - 1));

%% Sparse + Dense → dense
R4 = A + ones(3);
assert(~issparse(R4));
assert(isequal(R4, full(A) + ones(3)));

%% Dense + Sparse → dense
R5 = ones(3) + A;
assert(~issparse(R5));
assert(isequal(R5, ones(3) + full(A)));

%% Transpose
T = sparse([1 2 3], [2 3 1], [10 20 30], 3, 3);
Tt = T.';
assert(issparse(Tt));
assert(isequal(full(Tt), full(T).'));

%% Transpose of non-square
T2 = sparse([1 2], [1 3], [5 7], 2, 4);
T2t = T2.';
assert(issparse(T2t));
assert(isequal(size(T2t), [4 2]));
assert(isequal(full(T2t), full(T2).'));

%% Conjugate transpose (real sparse, same as transpose)
T3 = A';
assert(issparse(T3));
assert(isequal(full(T3), full(A)'));

%% Element-wise multiply (Sparse .* Sparse)
E1 = sparse([1 2 3], [1 2 3], [2 3 4], 3, 3);
E2 = sparse([1 2 3], [1 2 3], [10 20 30], 3, 3);
E3 = E1 .* E2;
assert(issparse(E3));
assert(isequal(full(E3), full(E1) .* full(E2)));

%% Element-wise multiply with different patterns
E4 = sparse([1 2], [1 2], [5 6], 3, 3);
E5 = sparse([2 3], [2 3], [7 8], 3, 3);
E6 = E4 .* E5;
assert(issparse(E6));
assert(nnz(E6) == 1);
assert(isequal(full(E6), full(E4) .* full(E5)));

%% Sparse .* scalar
E7 = E1 .* 10;
assert(issparse(E7));
assert(isequal(full(E7), full(E1) * 10));

%% Sparse ./ scalar
F1 = sparse([1 2], [1 2], [10 20], 2, 2);
F2 = F1 ./ 2;
assert(issparse(F2));
assert(isequal(full(F2), [5 0; 0 10]));

%% Comparison operators (densify)
G1 = sparse([1 2], [1 2], [3 4], 2, 2);
G2 = full(G1) == full(G1);
assert(all(G2(:)));

disp('SUCCESS')
