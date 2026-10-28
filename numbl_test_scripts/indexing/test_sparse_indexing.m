% Sparse matrices: reads keep results sparse, stores grow, deletion
% removes rows/columns (MATLAB rules). A one-element read is compared by
% value only (MATLAB returns a 1x1 sparse matrix, numbl a number).

S = sparse([1 2 3], [1 2 3], [4 5 6], 4, 4);
assert(full(S(2, 2)) == 5 && full(S(1, 2)) == 0 && full(S(6)) == 5);
b = S(:, 2);
assert(issparse(b) && isequal(size(b), [4 1]) && isequal(full(b), [0; 5; 0; 0]));
b = S([1 2], [2 3]);
assert(issparse(b) && isequal(full(b), [0 0; 5 0]));
b = S([2 1], :);
assert(isequal(full(b), [0 5 0 0; 4 0 0 0]));
b = S([3 1 3], [3 3]);
assert(isequal(full(b), [6 6; 0 0; 6 6]));
b = S([1 6 11]);
assert(issparse(b) && isequal(size(b), [1 3]) && isequal(full(b), [4 5 6]));
b = S([1; 6]);
assert(isequal(size(b), [2 1]));
b = S(:);
assert(issparse(b) && isequal(size(b), [16 1]) && nnz(b) == 3);
b = S(5:12);
assert(isequal(full(b), [0 5 0 0 0 0 6 0]));
b = S(logical(eye(4)));
assert(issparse(b) && isequal(size(b), [4 1]) && isequal(full(b), [4; 5; 6; 0]));
b = S(end, end);
assert(full(b) == 0);
b = S(2:3, end - 1);
assert(isequal(full(b), [0; 6]));
try
  b = S(5, 1); %#ok<NASGU>
  error('expected an error');
catch e
  assert(~strcmp(e.message, 'expected an error'));
end

% Stores
S(4, 4) = 7;
assert(nnz(S) == 4 && full(S(4, 4)) == 7);
S(2, 2) = 0;
assert(nnz(S) == 3);
S(5, 5) = 1;
assert(isequal(size(S), [5 5]) && nnz(S) == 4);
S([1 2], 1) = [8; 9];
assert(isequal(full(S(:, 1))', [8 9 0 0 0]));
S(:, 2) = 3;
assert(isequal(full(S(:, 2))', [3 3 3 3 3]));
S([2 2], 3) = [1; 2];
assert(full(S(2, 3)) == 2);
S(1:2, 4:5) = sparse([1 0; 0 2]);
assert(isequal(full(S(1:2, 4:5)), [1 0; 0 2]));
S2 = sparse(magic(3));
S2(:, :) = 0;
assert(nnz(S2) == 0 && isequal(size(S2), [3 3]));
T = sparse(1:3);
T(6) = 2;
assert(isequal(size(T), [1 6]) && isequal(full(T), [1 2 3 0 0 2]));

% Deletion
S = sparse([1 2 3], [1 2 3], [4 5 6], 4, 4);
S(:, 2) = [];
assert(issparse(S) && isequal(size(S), [4 3]) && nnz(S) == 2);
S(1, :) = [];
assert(isequal(size(S), [3 3]) && isequal(full(S), [0 0 0; 0 6 0; 0 0 0]));
T = sparse(1:3);
T(2) = [];
assert(isequal(size(T), [1 2]) && isequal(full(T), [1 3]));
C = sparse([1; 0; 2]);
C([1 2]) = [];
assert(isequal(size(C), [1 1]) && full(C) == 2);

disp('SUCCESS');
