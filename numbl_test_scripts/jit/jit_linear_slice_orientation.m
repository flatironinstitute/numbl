% Regression: a single-slot linear slice `a(1:n)` took its result
% orientation from a compile-time guess, and guessed ROW whenever the
% base's orientation was not statically provable.
%
% `isColVecTy` requires an *exact* leading dim, so a column built with a
% runtime-only length — `zeros(m, 1)` where `m` is not constant-folded —
% is not provably a column, and the slice came back transposed: `1 x n`
% instead of `n x 1`. The interpreter was right, so the two disagreed.
%
% Transposing a column is quiet rather than fatal: the values are all
% there and in order, so it only surfaces later, when the result meets
% another vector in an elementwise op and broadcasts to an n x n matrix
% instead of matching it. That is how it was found — a spherical-Hankel
% routine whose `j - 1i * y` silently became 5x5 instead of 5x1.
%
% Two details keep the buggy path reachable, and both matter:
%   - `ceil(x)` on a double argument keeps the length away from constant
%     folding; with an exact leading dim the base is provably a column
%     and the guess is never consulted;
%   - the struct-returning call below declines at the top level, which is
%     what pushed these functions down the compiled-callee path where the
%     bug lived.

function sz = slice_dyn_col(x, n)
  m = 3 + ceil(x);
  a = zeros(m, 1);          % column, runtime-only length
  a(1) = 1;
  sz = size(a(1:n));
end

function sz = slice_dyn_row(x, n)
  m = 3 + ceil(x);
  a = zeros(1, m);          % row, runtime-only length
  a(1) = 1;
  sz = size(a(1:n));
end

function sz = slice_dyn_mat(x, n)
  m = 3 + ceil(x);
  a = zeros(m, m);          % matrix: result follows the index, so a row
  a(1) = 1;
  sz = size(a(1:n));
end

% The failure mode that made the original bug visible: a transposed
% column broadcasts against another column instead of matching it.
function n = combine(x)
  m = 3 + ceil(x);
  a = zeros(m, 1);
  a(1) = 1;
  u = a(1:4);
  v = zeros(4, 1);
  v(1) = 1;
  n = numel(u - v);
end

function P = mkstruct()
  P.a = 1;
end

sc = slice_dyn_col(3.2, 3);
sr = slice_dyn_row(3.2, 3);
sm = slice_dyn_mat(3.2, 3);
nc = combine(3.2);
S = mkstruct();

assert(isequal(sc, [3 1]), 'dynamic column a(1:3) must stay a column');
assert(isequal(sr, [1 3]), 'dynamic row a(1:3) must stay a row');
assert(isequal(sm, [1 3]), 'matrix a(1:3) follows the index, so a row');
assert(nc == 4, 'column slice minus column must stay 4 elements, not broadcast');
assert(S.a == 1, 'struct helper');

disp('SUCCESS');
