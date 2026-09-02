% Random sparse matrices: sprand, sprandn, sprandsym.
rng(7);
% nnz is only approximately density*m*n (MATLAB samples with replacement)
S = sprand(50, 40, 0.1);
assert(issparse(S) && isequal(size(S), [50 40]));
assert(nnz(S) >= 120 && nnz(S) <= 220);
v = nonzeros(S);
assert(all(v > 0 & v < 1));

N = sprandn(30, 30, 0.2);
assert(issparse(N) && nnz(N) >= 110 && nnz(N) <= 200);
assert(any(nonzeros(N) < 0));

% same pattern as an existing matrix
P = sprand(N);
assert(isequal(P ~= 0, N ~= 0));

% symmetric
Y = sprandsym(40, 0.1);
assert(issparse(Y) && isequal(size(Y), [40 40]));
assert(norm(full(Y - Y'), 1) == 0);
assert(nnz(Y) > 0);

% empty pattern; density 1 fills most of the matrix (MATLAB samples with
% replacement, so it need not be full)
assert(nnz(sprand(5, 5, 0)) == 0);
assert(nnz(sprand(5, 5, 1)) >= 15);
disp('SUCCESS')
