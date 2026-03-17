% Test sparse matrix support — Phase 1: construction, conversion, queries

%% sparse(m, n) — zero matrix
S0 = sparse(3, 4);
assert(issparse(S0));
assert(isequal(size(S0), [3 4]));
assert(nnz(S0) == 0);
assert(numel(S0) == 12);
assert(length(S0) == 4);
assert(isempty(sparse(0, 5)));
assert(~isempty(S0));

%% sparse(i, j, v, m, n) — triplet construction
S1 = sparse([1 2 3], [1 2 3], [10 20 30], 3, 3);
assert(issparse(S1));
assert(isequal(size(S1), [3 3]));
assert(nnz(S1) == 3);
F1 = full(S1);
assert(~issparse(F1));
assert(isequal(F1, [10 0 0; 0 20 0; 0 0 30]));

%% sparse(i, j, v, m, n) — non-square
S2 = sparse([1 3], [2 4], [5 7], 3, 5);
assert(isequal(size(S2), [3 5]));
assert(nnz(S2) == 2);
F2 = full(S2);
expected = zeros(3, 5);
expected(1, 2) = 5;
expected(3, 4) = 7;
assert(isequal(F2, expected));

%% sparse(i, j, v) — inferred dimensions
S3 = sparse([2 4], [3 1], [9 8]);
assert(isequal(size(S3), [4 3]));
F3 = full(S3);
assert(F3(2, 3) == 9);
assert(F3(4, 1) == 8);
assert(nnz(S3) == 2);

%% Duplicate (i, j) entries are summed
S4 = sparse([1 1 2], [1 1 3], [5 3 7], 3, 3);
assert(nnz(S4) == 2);
F4 = full(S4);
assert(F4(1, 1) == 8);
assert(F4(2, 3) == 7);

%% sparse(A) — dense to sparse
A = [1 0 2; 0 0 3; 4 0 0];
SA = sparse(A);
assert(issparse(SA));
assert(nnz(SA) == 4);
assert(isequal(full(SA), A));

%% sparse(scalar)
S5 = sparse(42);
assert(issparse(S5));
assert(isequal(size(S5), [1 1]));
assert(nnz(S5) == 1);
assert(full(S5) == 42);

S6 = sparse(0);
assert(issparse(S6));
assert(nnz(S6) == 0);

%% full() passthrough for non-sparse
assert(isequal(full(5), 5));
assert(isequal(full([1 2 3]), [1 2 3]));

%% issparse returns false for non-sparse
assert(~issparse(42));
assert(~issparse([1 2 3]));
assert(~issparse('hello'));

%% class() returns 'double' for sparse
assert(isequal(class(S1), 'double'));

%% isnumeric and isfloat
assert(isnumeric(S1));
assert(isfloat(S1));

%% sparse(i, j, v, m, n, nzmax) — nzmax ignored
S7 = sparse([1 2], [1 2], [3 4], 3, 3, 100);
assert(isequal(full(S7), [3 0 0; 0 4 0; 0 0 0]));

%% Triplet with scalar v (broadcast)
S8 = sparse([1 2 3], [1 2 3], 1, 4, 4);
assert(nnz(S8) == 3);
F8 = full(S8);
assert(F8(1,1) == 1);
assert(F8(2,2) == 1);
assert(F8(3,3) == 1);
assert(F8(4,4) == 0);

%% Empty sparse
S9 = sparse(0, 0);
assert(issparse(S9));
assert(isempty(S9));
assert(isequal(size(S9), [0 0]));

%% sparse(sparse) is passthrough
S10 = sparse(S1);
assert(issparse(S10));

%% Multiple entries in same column, different rows
S11 = sparse([1 3 2], [2 2 2], [10 30 20], 3, 3);
assert(nnz(S11) == 3);
F11 = full(S11);
assert(F11(1,2) == 10);
assert(F11(2,2) == 20);
assert(F11(3,2) == 30);

disp('SUCCESS')
