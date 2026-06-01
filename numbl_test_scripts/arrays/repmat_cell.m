% Test: repmat on cell arrays (column-major tiling), like any other array
% type. Verified against MATLAB R2025b.

% Replicate a 1x1 cell into a column (the surfaceop updateRHS pattern:
% repmat({f}, n, 1) for a function handle f).
f = @(x, y, z) sin(x .* y);
c = repmat({f}, 3, 1);
assert(isequal(size(c), [3 1]), 'repmat({f},3,1) is 3x1');
assert(isa(c{1}, 'function_handle'), 'element is a function handle');
assert(c{2}(2, 3, 0) == sin(6), 'replicated handle still callable');

% 2-D tiling of a 1x2 cell.
d = repmat({1, 'a'}, 2, 3);
assert(isequal(size(d), [2 6]), 'repmat({1,''a''},2,3) is 2x6');
assert(isequal(d{1, 1}, 1), 'tile (1,1) value');
assert(isequal(d{2, 2}, 'a'), 'tile wraps columns');
assert(isequal(d{1, 3}, 1), 'tile wraps to source col 1');

% Single rep argument replicates in both dimensions.
e = repmat({7}, 2);
assert(isequal(size(e), [2 2]), 'repmat({7},2) is 2x2');
assert(isequal(e{2, 2}, 7), 'all entries equal source');

disp('SUCCESS')
