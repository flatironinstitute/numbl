% Test isequal with sparse matrices

%% isequal(sparse, sparse) — same
S = sparse([1 0; 0 2]);
assert(isequal(S, S));

%% isequal(sparse, sparse) — different
S2 = sparse([1 0; 0 3]);
assert(~isequal(S, S2));

%% isequal(sparse, dense)
assert(isequal(S, full(S)));
assert(isequal(full(S), S));

%% isequal(complex sparse, complex sparse)
CS = sparse([1+2i 0; 0 3+4i]);
assert(isequal(CS, CS));

%% isequal(complex sparse, complex dense)
assert(isequal(CS, full(CS)));

%% isequal with different sizes
S3 = sparse([1 0 0; 0 2 0]);
assert(~isequal(S, S3));

disp('SUCCESS')
