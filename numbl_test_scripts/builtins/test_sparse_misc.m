% Test miscellaneous sparse matrix operations

%% isreal on complex sparse
CS = sparse([1+2i 0; 0 3]);
assert(~isreal(CS));

%% isreal on real sparse
S = sparse([1 0; 0 2]);
assert(isreal(S));

%% sign on sparse
S2 = sparse([3 0; 0 -4]);
R = sign(S2);
assert(isequal(R, sign(full(S2))));

%% repmat sparse
S3 = sparse([1 0; 0 2]);
R2 = repmat(S3, 2, 1);
expected = repmat(full(S3), 2, 1);
assert(isequal(R2, expected));

%% repmat sparse horizontal
R3 = repmat(S3, 1, 3);
expected3 = repmat(full(S3), 1, 3);
assert(isequal(R3, expected3));

%% sparse & dense (element-wise logical AND)
D = [1 1; 1 1];
R4 = S3 & D;
expected4 = full(S3) & D;
assert(isequal(R4, expected4));

%% sparse | dense (element-wise logical OR)
R5 = S3 | D;
expected5 = full(S3) | D;
assert(isequal(R5, expected5));

disp('SUCCESS')
