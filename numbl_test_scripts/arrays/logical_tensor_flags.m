% Test: logical tensor _isLogical flag is set correctly by builtins that
% return logical arrays (isnan, isinf, isfinite, logical(), true(m,n),
% false(m,n)).  Without the flag, these masks fail to do logical indexing.

% ── isnan ────────────────────────────────────────────────────────────────
x = [1, NaN, 3, NaN];
mask = isnan(x);
assert(numel(x(mask)) == 2);          % two NaN elements selected
assert(all(isnan(x(mask))));          % all selected are NaN
x(mask) = 0;
assert(isequal(x, [1, 0, 3, 0]));    % NaN slots zeroed out

% ── isinf ────────────────────────────────────────────────────────────────
y = [1, Inf, -Inf, 4];
assert(numel(y(isinf(y))) == 2);
assert(all(isinf(y(isinf(y)))));

% ── isfinite ─────────────────────────────────────────────────────────────
assert(isequal(y(isfinite(y)), [1, 4]));

% ── logical(vector) ──────────────────────────────────────────────────────
v = logical([1, 0, 2, 0]);
assert(isequal(size(v), [1, 4]));
data = [10, 20, 30, 40];
assert(isequal(data(v), [10, 30]));

% ── true(m,n) / false(m,n) ───────────────────────────────────────────────
T = true(2, 3);
assert(isequal(size(T), [2, 3]));
assert(all(T(:)));

F = false(2, 3);
assert(isequal(size(F), [2, 3]));
assert(~any(F(:)));

% true(n) → n×n
T2 = true(3);
assert(isequal(size(T2), [3, 3]));

% ── min/max of logical tensors ───────────────────────────────────────────
L = logical([1, 0, 1, 0]);
assert(isequal(min(L), false));
assert(isequal(max(L), true));

M = logical([0, 1; 1, 0]);
minM = min(M);                  % column-wise min: [0, 0]
assert(isequal(size(minM), [1, 2]));
assert(isequal(minM, logical([0, 0])));

% min/max along dim 2 produces a logical column vector usable as a mask
data2 = [10, 20];
maxM2 = max(M, [], 2);         % row-wise max: [1; 1]
assert(isequal(maxM2, logical([1; 1])));
assert(isequal(data2(maxM2), [10, 20]));  % logical mask selects both elements

disp('SUCCESS')
