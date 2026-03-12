% Test uniquetol builtin
% ByRows with explicit tolerance
A = [1 2; 1.001 2.001; 3 4];
[C, ~, ic] = uniquetol(A, 0.01, 'ByRows', true);
assert(size(C, 1) == 2);  % rows 1 and 2 merged
assert(size(C, 2) == 2);
assert(ic(1) == ic(2));    % rows 1 and 2 map to same unique row
assert(ic(3) ~= ic(1));   % row 3 is different

% Rows that are NOT within tolerance stay separate
B = [1 2; 1.1 2.1; 3 4];
[C2, ~, ic2] = uniquetol(B, 0.01, 'ByRows', true);
assert(size(C2, 1) == 3);  % all rows are unique (0.1 > tol=0.01)

% Larger tolerance merges more
[C3, ~, ic3] = uniquetol(B, 0.2, 'ByRows', true);
assert(size(C3, 1) == 2);  % rows 1 and 2 within tol=0.2

fprintf('SUCCESS\n');
