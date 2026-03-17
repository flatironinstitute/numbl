% Test complex sparse matrix support

%% sparse(A) with complex dense input
A = [1+2i 0; 0 3-4i];
S = sparse(A);
assert(issparse(S));
assert(nnz(S) == 2);
assert(isequal(full(S), A));

%% sparse(complex_scalar)
S1 = sparse(2+3i);
assert(issparse(S1));
F1 = full(S1);
assert(real(F1) == 2);
assert(imag(F1) == 3);

%% sparse(0) complex — note: 0+0i is just 0 in MATLAB
S2 = sparse(0);
assert(issparse(S2));
assert(nnz(S2) == 0);

%% full() preserves complex
B = [0 1i; 2 0];
SB = sparse(B);
assert(isequal(full(SB), B));

%% Complex sparse + complex sparse
C1 = sparse([1 2], [1 2], [1+1i 2+2i], 2, 2);
C2 = sparse([1 2], [1 2], [3+3i 4+4i], 2, 2);
C3 = C1 + C2;
assert(issparse(C3));
assert(isequal(full(C3), full(C1) + full(C2)));

%% Complex sparse - complex sparse
C4 = C1 - C2;
assert(issparse(C4));
assert(isequal(full(C4), full(C1) - full(C2)));

%% Negation of complex sparse
C5 = -C1;
assert(issparse(C5));
assert(isequal(full(C5), -full(C1)));

%% Complex sparse * real scalar
C6 = C1 * 3;
assert(issparse(C6));
assert(isequal(full(C6), full(C1) * 3));

%% Complex sparse * complex scalar
C7 = C1 * (1+1i);
assert(issparse(C7));
assert(isequal(full(C7), full(C1) * (1+1i)));

%% Complex sparse * complex sparse (matmul)
M1 = sparse([1 1 2], [1 2 2], [1+1i 2 3i], 2, 2);
M2 = sparse([1 2], [1 2], [1i 2-1i], 2, 2);
M3 = M1 * M2;
assert(issparse(M3));
assert(isequal(full(M3), full(M1) * full(M2)));

%% Transpose of complex sparse (non-conjugate)
T1 = sparse([1 2], [1 2], [1+2i 3+4i], 2, 2);
T2 = T1.';
assert(issparse(T2));
assert(isequal(full(T2), full(T1).'));

%% Conjugate transpose of complex sparse
T3 = T1';
assert(issparse(T3));
assert(isequal(full(T3), full(T1)'));

%% Complex sparse .* complex sparse
E1 = sparse([1 2], [1 2], [1+1i 2-1i], 2, 2);
E2 = sparse([1 2], [1 2], [2+1i 1+3i], 2, 2);
E3 = E1 .* E2;
assert(issparse(E3));
assert(isequal(full(E3), full(E1) .* full(E2)));

%% Complex sparse ./ scalar
D1 = C1 ./ 2;
assert(issparse(D1));
assert(isequal(full(D1), full(C1) ./ 2));

%% Real sparse + complex sparse
R1 = sparse([1 2], [1 2], [10 20], 2, 2);
C8 = R1 + C1;
assert(issparse(C8));
assert(isequal(full(C8), full(R1) + full(C1)));

%% Sparse * complex dense (matmul)
SD = C1 * eye(2);
assert(~issparse(SD));
assert(isequal(SD, full(C1)));

%% Dense * complex sparse (matmul)
DS = eye(2) * C1;
assert(~issparse(DS));
assert(isequal(DS, full(C1)));

disp('SUCCESS')
