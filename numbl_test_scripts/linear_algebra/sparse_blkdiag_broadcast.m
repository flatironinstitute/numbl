% Test: blkdiag with sparse blocks, and sparse .* with implicit expansion
% (a column/row/scalar broadcast against a sparse matrix). Verified against
% MATLAB R2025b.

% --- blkdiag accepts sparse blocks ---
A = blkdiag(speye(2), 3 * speye(3));
assert(isequal(size(A), [5 5]), 'blkdiag size 5x5');
assert(isequal(full(A), blkdiag(eye(2), 3 * eye(3))), 'blkdiag sparse == dense');

% Mixed sparse and dense blocks.
B = blkdiag(speye(2), [1 2; 3 4]);
assert(isequal(full(B), blkdiag(eye(2), [1 2; 3 4])), 'mixed blkdiag');

% --- sparse .* broadcasting ---
S = speye(3);
c = [10; 20; 30];           % column vector
R = c .* S;                  % implicit expansion -> diag(c)
assert(isequal(full(R), diag([10 20 30])), 'column .* sparse');

r = [1 2 3];                 % row vector
R2 = S .* r;                 % columns scaled
assert(isequal(full(R2), diag([1 2 3])), 'sparse .* row');

R3 = 2 .* S;                 % scalar
assert(isequal(full(R3), 2 * eye(3)), 'scalar .* sparse');

% Broadcasting against a non-diagonal sparse matrix.
M = sparse([1 1 2], [1 2 3], [5 6 7], 2, 3); % full: [5 6 0; 0 0 7]
cc = [2; 3];
RM = cc .* M;
assert(isequal(full(RM), [10 12 0; 0 0 21]), 'column .* general sparse');

disp('SUCCESS')
