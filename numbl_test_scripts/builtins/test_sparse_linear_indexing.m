% Test sparse matrix linear vector indexing

%% S(vector) — linear indexing with numeric vector
S = sparse([1 0; 0 2]);
R = S([1 4]);
% Linear index 1 = (1,1) = 1, index 4 = (2,2) = 2
assert(isequal(full(R), [1 2]));

%% S(vector) with zeros in result
R2 = S([1 2 3 4]);
assert(isequal(full(R2), [1 0 0 2]));

%% Complex sparse linear indexing
CS = sparse([1+2i 0; 0 3+4i]);
R3 = CS([1 4]);
expected = [1+2i 3+4i];
assert(isequal(full(R3), expected));

%% Logical indexing
idx = logical([1 0 0 1]);
R4 = S(idx);
assert(isequal(full(R4), [1 2]));

disp('SUCCESS')
