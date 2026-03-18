% Test builtin functions on sparse matrices

%% abs on real sparse
S = sparse([3 0; 0 -4]);
R = abs(S);
assert(isequal(R, abs(full(S))));

%% abs on complex sparse
CS = sparse([1+2i 0; 0 3-4i]);
R2 = abs(CS);
assert(max(max(abs(R2 - abs(full(CS))))) < 1e-10);

%% real on complex sparse
R3 = real(CS);
expected3 = real(full(CS));
assert(isequal(R3, expected3));

%% imag on complex sparse
R4 = imag(CS);
expected4 = imag(full(CS));
assert(isequal(R4, expected4));

%% conj on complex sparse
R5 = conj(CS);
expected5 = conj(full(CS));
assert(isequal(R5, expected5));

%% max on sparse
S2 = sparse([3 0; 0 4]);
R6 = max(S2);
assert(isequal(full(R6), max(full(S2))));

%% min on sparse
R7 = min(S2);
assert(isequal(full(R7), min(full(S2))));

%% logical not on sparse
R8 = ~S2;
expected8 = ~full(S2);
assert(isequal(R8, expected8));

%% norm on sparse
R_norm = norm(S2);
assert(R_norm == norm(full(S2)));

%% sqrt on sparse
R9 = sqrt(S2);
assert(isequal(R9, sqrt(full(S2))));

%% floor/ceil on sparse
SF = sparse([1.5 0; 0 2.7]);
assert(isequal(floor(SF), floor(full(SF))));
assert(isequal(ceil(SF), ceil(full(SF))));

disp('SUCCESS')
