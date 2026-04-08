% Test two related indexing fixes:
%
% 1. Indexing a logical array preserves the logical class, so it can be
%    used as a logical mask downstream (e.g. q(qrem(q)) pattern used by
%    FLAM's ifmm_mv for permutation bookkeeping).
%
% 2. Assigning into an empty cell element with a colon index grows the
%    cell element from empty using the RHS's dimensions (e.g.
%    Z{k}(1:n, :) = M when Z{k} is empty).

% --- Part 1: logical preservation through indexing ---

qrem = logical([1 0 1 1]);

% Indexing a logical with a numeric vector must return logical.
y = qrem([2 1 3 4]);
assert(islogical(y), 'qrem(vec) should be logical');
assert(isequal(y, logical([0 1 1 1])), 'qrem(vec) wrong values');

% Scalar element of a logical is logical.
e = qrem(1);
assert(islogical(e), 'qrem(1) should be logical');
assert(e == true, 'qrem(1) == true');

% 2-D logical indexing preserves logical class.
L = logical([1 0; 1 1]);
row = L(1, :);
assert(islogical(row), 'L(1,:) logical');
col = L(:, 2);
assert(islogical(col), 'L(:,2) logical');

% The chunkie/FLAM trigger: q(qrem(q)) where the inner index is a
% logical picks elements of q through a logical mask.
q = [2 1 3 4];
sel = q(qrem(q));
assert(isequal(sel, [1 3 4]), 'q(qrem(q)) should select via logical mask');

% --- Part 2: cell element grow-from-empty with colon index ---

Z = cell(3, 1);
Z{2}(1:3, :) = ones(3, 4);
assert(isequal(size(Z{2}), [3 4]), 'grow 2d cell element');
assert(all(Z{2}(:) == 1), 'grow 2d cell values');

% Row-colon grow
W = cell(1, 1);
W{1}(:, 1:5) = reshape(1:10, 2, 5);
assert(isequal(size(W{1}), [2 5]), 'col-indexed grow');
assert(W{1}(2, 3) == 6, 'col-indexed value');

disp('SUCCESS');
