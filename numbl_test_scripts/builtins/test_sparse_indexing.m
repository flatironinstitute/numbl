% Test sparse matrix indexing, find, nonzeros, and index assignment

%% Setup
S = sparse([1 2 3 1], [1 2 3 3], [10 20 30 5], 3, 4);
% S is 3x4 with: (1,1)=10, (2,2)=20, (3,3)=30, (1,3)=5

%% Element access S(i,j) — nonzero element
v = S(1,1);
assert(full(v) == 10);
assert(issparse(v));

%% Element access S(i,j) — zero element
v0 = S(1,2);
assert(full(v0) == 0);
assert(issparse(v0));

%% Element access S(i,j) — another nonzero
v2 = S(1,3);
assert(full(v2) == 5);

%% Column extraction S(:,j)
col2 = S(:,2);
assert(issparse(col2));
F2 = full(col2);
assert(isequal(F2, [0; 20; 0]));

%% Column extraction — column with multiple nonzeros
col3 = S(:,3);
assert(issparse(col3));
F3 = full(col3);
assert(isequal(F3, [5; 0; 30]));

%% Column extraction — empty column
col4 = S(:,4);
assert(issparse(col4));
assert(nnz(col4) == 0);

%% Row extraction S(i,:)
row1 = S(1,:);
assert(issparse(row1));
FR1 = full(row1);
assert(isequal(FR1, [10 0 5 0]));

%% Row extraction — row with single nonzero
row2 = S(2,:);
assert(issparse(row2));
FR2 = full(row2);
assert(isequal(FR2, [0 20 0 0]));

%% Submatrix extraction S([i1,i2],[j1,j2])
sub = S([1 2], [1 2]);
assert(issparse(sub));
assert(isequal(full(sub), [10 0; 0 20]));

%% Submatrix — larger
sub2 = S([1 3], [1 3 4]);
assert(issparse(sub2));
assert(isequal(full(sub2), [10 5 0; 0 30 0]));

%% Linear indexing S(k) — single element
v_lin = S(1);
assert(full(v_lin) == 10);
assert(issparse(v_lin));

%% Linear indexing — zero element
v_lin0 = S(2);
assert(full(v_lin0) == 0);

%% S(:) — reshape to column
Scol = S(:);
assert(issparse(Scol));
sz = size(Scol);
assert(sz(1) == 12);
assert(sz(2) == 1);
assert(nnz(Scol) == 4);

%% find with 3 outputs [i,j,v]
[fi, fj, fv] = find(S);
assert(isequal(fi, [1; 2; 1; 3]));
assert(isequal(fj, [1; 2; 3; 3]));
assert(isequal(fv, [10; 20; 5; 30]));

%% find with 1 output — linear indices
idx = find(S);
% Column-major linear indices: (1,1)=1, (2,2)=5, (1,3)=7, (3,3)=9
assert(isequal(idx, [1; 5; 7; 9]));

%% find with 2 outputs [i,j]
[fi2, fj2] = find(S);
assert(isequal(fi2, [1; 2; 1; 3]));
assert(isequal(fj2, [1; 2; 3; 3]));

%% find with count limit
[fi3, fj3, fv3] = find(S, 2);
assert(isequal(fi3, [1; 2]));
assert(isequal(fj3, [1; 2]));
assert(isequal(fv3, [10; 20]));

%% find on empty sparse
S_empty = sparse(3, 4);
idx_empty = find(S_empty);
assert(isempty(idx_empty));

%% nonzeros
nz = nonzeros(S);
assert(isequal(nz, [10; 20; 5; 30]));

%% nonzeros on empty sparse
nz_empty = nonzeros(S_empty);
assert(isempty(nz_empty));

%% Index assignment — set zero to nonzero
S3 = S;
S3(1,2) = 99;
assert(issparse(S3));
assert(full(S3(1,2)) == 99);
assert(nnz(S3) == 5);

%% Index assignment — overwrite existing nonzero
S4 = S;
S4(1,1) = 77;
assert(full(S4(1,1)) == 77);
assert(nnz(S4) == 4);

%% Index assignment — set nonzero to zero (removes entry)
S5 = S;
S5(1,1) = 0;
assert(full(S5(1,1)) == 0);
assert(nnz(S5) == 3);

%% Complex sparse indexing
CS = sparse([1 2], [1 2], [1+2i 3+4i], 2, 2);
cv = CS(1,1);
assert(full(cv) == 1+2i);

%% Complex sparse find
[ci, cj, cv2] = find(CS);
assert(isequal(ci, [1; 2]));
assert(isequal(cj, [1; 2]));
assert(isequal(cv2, [1+2i; 3+4i]));

%% Complex sparse nonzeros
cnz = nonzeros(CS);
assert(isequal(cnz, [1+2i; 3+4i]));

%% size/numel on indexed sparse
assert(isequal(size(S(:,2)), [3 1]));
assert(isequal(size(S(1,:)), [1 4]));

disp('SUCCESS')
