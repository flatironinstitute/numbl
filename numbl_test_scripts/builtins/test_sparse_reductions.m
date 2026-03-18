% Test reduction operations on sparse matrices

%% prod along columns
S = sparse([1 2; 3 4]);
R = prod(S);
assert(isequal(R, prod(full(S))));

%% prod along rows
R2 = prod(S, 2);
assert(isequal(R2, prod(full(S), 2)));

%% cumsum
R3 = cumsum(S);
assert(isequal(R3, cumsum(full(S))));

%% cumprod
R4 = cumprod(S);
assert(isequal(R4, cumprod(full(S))));

%% diff
R5 = diff(S);
assert(isequal(R5, diff(full(S))));

disp('SUCCESS')
