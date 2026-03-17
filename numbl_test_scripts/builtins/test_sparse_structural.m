% Test sparse structural operations: speye, spdiags, diag, reshape,
% horzcat, vertcat, sum, any, all

%% speye — square
S1 = speye(3);
assert(issparse(S1));
assert(isequal(full(S1), eye(3)));

%% speye — rectangular
S2 = speye(2, 4);
assert(issparse(S2));
assert(isequal(full(S2), eye(2, 4)));

%% speye — tall
S2b = speye(4, 2);
assert(issparse(S2b));
assert(isequal(full(S2b), eye(4, 2)));

%% speye(1)
S1x1 = speye(1);
assert(issparse(S1x1));
assert(full(S1x1) == 1);

%% diag — extract diagonal from sparse matrix
S3 = sparse([1 2 3], [1 2 3], [10 20 30], 3, 3);
d = diag(S3);
assert(issparse(d));
assert(isequal(full(d), [10; 20; 30]));

%% diag — extract with offset
S3b = sparse([1 1 2 3], [1 2 2 3], [10 5 20 30], 3, 3);
d1 = diag(S3b, 1);
assert(issparse(d1));
assert(isequal(full(d1), [5; 0]));

%% diag — create diagonal sparse from sparse vector
v = sparse([1 3], [1 1], [10 30], 3, 1);
D = diag(v);
assert(issparse(D));
expected = [10 0 0; 0 0 0; 0 0 30];
assert(isequal(full(D), expected));

%% reshape sparse
S4 = sparse([1 2], [1 2], [10 20], 2, 3);
R = reshape(S4, 3, 2);
assert(issparse(R));
assert(isequal(full(R), [10 20; 0 0; 0 0]));

%% reshape sparse — same shape (no-op)
R2 = reshape(S4, 2, 3);
assert(issparse(R2));
assert(isequal(full(R2), full(S4)));

%% horzcat sparse + sparse
A1 = sparse([1], [1], [5], 2, 2);
A2 = sparse([2], [1], [7], 2, 1);
H = [A1 A2];
assert(issparse(H));
assert(isequal(full(H), [5 0 0; 0 0 7]));

%% vertcat sparse + sparse
V = [A1; sparse([1], [2], [9], 1, 2)];
assert(issparse(V));
assert(isequal(full(V), [5 0; 0 0; 0 9]));

%% horzcat sparse + dense → sparse
S5 = sparse([1 2 3], [1 2 3], [10 20 30], 3, 3);
M = [S5 eye(3)];
assert(issparse(M));
assert(isequal(full(M), [10 0 0 1 0 0; 0 20 0 0 1 0; 0 0 30 0 0 1]));

%% vertcat sparse + dense → sparse
V2 = [S5; eye(3)];
assert(issparse(V2));
assert(isequal(full(V2), [10 0 0; 0 20 0; 0 0 30; 1 0 0; 0 1 0; 0 0 1]));

%% sum — default (along dim 1)
S6 = sparse([1 2 3 1], [1 2 3 3], [10 20 30 5], 3, 3);
s1 = sum(S6);
assert(issparse(s1));
assert(isequal(full(s1), [10 20 35]));

%% sum along dim 2
s2 = sum(S6, 2);
assert(isequal(full(s2), [15; 20; 30]));

%% sum of empty sparse
Se = sparse(3, 3);
s3 = sum(Se);
assert(isequal(full(s3), [0 0 0]));

%% any — default (along dim 1)
a1 = any(S6);
assert(isequal(full(a1), [1 1 1]));

%% any along dim 2
a2 = any(S6, 2);
assert(isequal(full(a2), [1; 1; 1]));

%% any on empty sparse
a3 = any(Se);
assert(isequal(full(a3), [0 0 0]));

%% all — default (along dim 1)
al1 = all(S6);
assert(isequal(full(al1), [0 0 0]));

%% all on full-column sparse
S7 = sparse([1 2 3 1 2 3], [1 1 1 2 2 2], [1 2 3 4 5 6], 3, 2);
al2 = all(S7);
assert(isequal(full(al2), [1 1]));

%% Multiple sparse horzcat
H2 = [speye(2) speye(2) speye(2)];
assert(issparse(H2));
assert(isequal(size(H2), [2 6]));

%% spdiags — construct diagonal sparse matrix
B = [1 4; 2 5; 3 6];
d2 = [0 1];
S8 = spdiags(B, d2, 3, 3);
assert(issparse(S8));
assert(isequal(full(S8), [1 5 0; 0 2 6; 0 0 3]));

disp('SUCCESS')
