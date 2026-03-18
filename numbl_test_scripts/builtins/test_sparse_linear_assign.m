% Test linear indexing assignment on sparse matrices

%% S(k) = val — single linear index
S = sparse([1 0; 0 2]);
S(2) = 5;
expected = [1 0; 5 2];
assert(isequal(full(S), expected));

%% S(k) = 0 — remove element
S2 = sparse([1 0; 0 2]);
S2(1) = 0;
assert(nnz(S2) == 1);

disp('SUCCESS')
