% Logical-array indexing after e2 per-assign kernels.
%
% Comparison ops (>, <, ==, etc.) and unary ! produce logical arrays.
% Under --opt e2 those arrays are JIT-compiled; the resulting RuntimeTensor
% must carry _isLogical=true so that subsequent indexing treats the values
% as a boolean mask (not as integer column indices). Without the fix,
% pts(:, mask) fails with "Index exceeds array bounds" because column 0
% does not exist.

n = 6000;
x = linspace(-2, 2, n);

% ── Test 1: simple comparison mask ──────────────────────────────────────
mask = x > 0;
selected = x(mask);
assert(all(selected > 0), 'test1: selected values should all be > 0');
expected_count = sum(x > 0);
assert(numel(selected) == expected_count, ...
    sprintf('test1: count mismatch %d vs %d', numel(selected), expected_count));

% ── Test 2: 2D column selection (the chunkie pattern) ───────────────────
pts = [x; x .* 0.5];          % 2 x n matrix
mask2 = x > 0.5;
sub = pts(:, mask2);
assert(size(sub, 1) == 2, 'test2: row count should be 2');
assert(all(sub(1,:) > 0.5), 'test2: all selected pts should have x > 0.5');

% ── Test 3: min(abs(...)) > tol — the exact chunkie iffy pattern ────────
a = linspace(-3, 3, n);
tol = 1e-2;
iffy = min(abs(a + 1), abs(a)) > tol;
ipt = find(iffy);
assert(isnumeric(ipt), 'test3: find result should be numeric');
% Points where min-distance-to-0-or-(-1) > tol: both sides away from the
% singularities at 0 and -1. Quick sanity: none of the "iffy" positions
% should be within tol of 0 or -1.
a_iffy = a(iffy);
assert(all(abs(a_iffy) > tol & abs(a_iffy + 1) > tol), ...
    'test3: iffy values should not be near 0 or -1');

% ── Test 4: chained comparisons and logical ops ─────────────────────────
y = sin(x);
mask4 = y > 0.5;
mask5 = x > 0;
both = x(mask4 & mask5);
assert(all(both > 0), 'test4: x values should be > 0');
assert(all(sin(both) > 0.5), 'test4: sin(x) values should be > 0.5');

% ── Test 5: unary not ───────────────────────────────────────────────────
pos = x > 0;
non_pos = ~pos;
assert(all(x(non_pos) <= 0), 'test5: negated mask should select x <= 0');
assert(sum(pos) + sum(non_pos) == n, 'test5: pos + non_pos should cover all n');

disp('SUCCESS')
