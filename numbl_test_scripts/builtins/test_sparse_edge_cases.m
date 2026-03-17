% Test sparse edge cases found during code audit

%% 1. Submatrix with duplicate row indices: S([1 1], :)
S1 = sparse([1 2 3], [1 2 3], [10 20 30], 3, 3);
R1 = S1([1 1], :);
assert(isequal(size(R1), [2 3]));
assert(isequal(full(R1), [10 0 0; 10 0 0]));

%% 2. Submatrix with duplicate col indices: S(:, [2 2])
R2 = S1(:, [2 2]);
assert(isequal(size(R2), [3 2]));
assert(isequal(full(R2), [0 0; 20 20; 0 0]));

%% 3. diag of sparse row vector
v_row = sparse([1 1], [1 3], [10 30], 1, 3);
D_row = diag(v_row);
assert(issparse(D_row));
assert(isequal(full(D_row), diag([10 0 30])));

%% 4. diag of sparse column vector
v_col = sparse([1 3], [1 1], [10 30], 3, 1);
D_col = diag(v_col);
assert(issparse(D_col));
assert(isequal(full(D_col), diag([10; 0; 30])));

%% 5. diag with negative offset on sparse vector
v2 = sparse([1 2], [1 1], [5 7], 2, 1);
D_neg = diag(v2, -1);
assert(issparse(D_neg));
assert(isequal(full(D_neg), diag([5; 7], -1)));

%% 6. diag with positive offset on sparse vector
D_pos = diag(v2, 1);
assert(issparse(D_pos));
assert(isequal(full(D_pos), diag([5; 7], 1)));

%% 7. horzcat of multiple sparse matrices
A = sparse([1], [1], [1], 2, 1);
B = sparse([2], [1], [2], 2, 1);
C = sparse([1], [1], [3], 2, 1);
H = [A B C];
assert(issparse(H));
assert(isequal(full(H), [1 0 3; 0 2 0]));

%% 8. vertcat of multiple sparse matrices
V = [A'; B'; C'];
assert(issparse(V));
assert(isequal(full(V), [1 0; 0 2; 3 0]));

%% 9. reshape sparse column to row
S2 = sparse([1 3], [1 1], [10 30], 3, 1);
R3 = reshape(S2, 1, 3);
assert(issparse(R3));
assert(isequal(full(R3), [10 0 30]));

%% 10. reshape sparse 2x3 to 3x2
S3 = sparse([1 2 1], [1 2 3], [1 2 3], 2, 3);
R4 = reshape(S3, 3, 2);
assert(issparse(R4));
assert(isequal(full(R4), full(reshape(full(S3), 3, 2))));

%% 11. reshape sparse 2x3 to 6x1
R5 = reshape(S3, 6, 1);
assert(issparse(R5));
assert(isequal(full(R5), full(reshape(full(S3), 6, 1))));

%% 12. flipud on sparse
S4 = sparse([1 3], [1 2], [10 30], 3, 2);
F1 = flipud(S4);
assert(issparse(F1));
assert(isequal(full(F1), flipud(full(S4))));

%% 13. fliplr on sparse
F2 = fliplr(S4);
assert(issparse(F2));
assert(isequal(full(F2), fliplr(full(S4))));

%% 14. sparse * ones (sparse times dense)
S5 = sparse([1 2], [1 2], [3 4], 2, 2);
R6 = S5 * ones(2, 1);
assert(isequal(R6, [3; 4]));

%% 15. find with 'last' direction on sparse
S6 = sparse([1 2 3], [1 2 3], [10 20 30], 3, 3);
[i1, j1, v1] = find(S6, 2, 'last');
assert(isequal(i1, [2; 3]));
assert(isequal(j1, [2; 3]));
assert(isequal(v1, [20; 30]));

%% 16. sum of 1x1 sparse
S7 = sparse(1, 1, 5, 1, 1);
assert(full(sum(S7)) == 5);

%% 17. triu with offset on sparse
S8 = sparse([1 2 3 1 2 3], [1 1 1 2 2 3], [1 2 3 4 5 6], 3, 3);
T1 = triu(S8, 1);
assert(issparse(T1));
assert(isequal(full(T1), triu(full(S8), 1)));

%% 18. tril with offset on sparse
T2 = tril(S8, -1);
assert(issparse(T2));
assert(isequal(full(T2), tril(full(S8), -1)));

%% 19. spdiags construct then verify diag extract
B = [1 4; 2 5; 3 6];
S9 = spdiags(B, [0 1], 3, 3);
d0 = diag(S9);
d1 = diag(S9, 1);
assert(isequal(full(d0), [1; 2; 3]));
assert(isequal(full(d1), [5; 6]));

%% 20. Sparse index assignment — overwrite then remove
S10 = sparse([1 2], [1 2], [10 20], 3, 3);
S10(1,1) = 99;
assert(full(S10(1,1)) == 99);
S10(1,1) = 0;
assert(full(S10(1,1)) == 0);
assert(nnz(S10) == 1);

%% 21. speye rectangular with more cols than rows
S11 = speye(2, 5);
assert(issparse(S11));
assert(isequal(full(S11), eye(2, 5)));

%% 22. sparse horzcat with dense scalar
% In MATLAB [sparse_mat, 5] errors — skip if not valid
% Instead test horzcat sparse + dense matrix
H2 = [speye(2) eye(2)];
assert(issparse(H2));
assert(isequal(full(H2), [eye(2) eye(2)]));

disp('SUCCESS')
