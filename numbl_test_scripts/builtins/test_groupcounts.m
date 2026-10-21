% groupcounts on vectors, matrices and cellstr columns
[c, g, p] = groupcounts([3 1 3 NaN 2 NaN 3]');
assert(isequal(c, [1; 1; 3; 2]));
assert(isequal(g(1:3), [1; 2; 3]) && isnan(g(4)) && numel(g) == 4);
assert(max(abs(p - 100 * [1; 1; 3; 2] / 7)) < 1e-12);

% the fmm3dbie idiom
norders = [4 4 6; 4 6 6];
[nc, nv] = groupcounts(norders(:));
assert(isequal(nc, [3; 3]) && isequal(nv, [4; 6]));

c = groupcounts([2; 2; 5]);
assert(isequal(c, [2; 1]));

[c, g] = groupcounts([true; false; true]);
assert(isequal(c, [1; 2]) && islogical(g) && isequal(g, [false; true]));

[c, g] = groupcounts([-0; 0; Inf; -Inf]);
assert(isequal(c, [1; 2; 1]) && isequal(g, [-Inf; 0; Inf]));

[c, g] = groupcounts(zeros(0, 1));
assert(isequal(size(c), [0 1]) && isequal(size(g), [0 1]));

% each column of a matrix is a grouping variable
[c, g] = groupcounts([2 NaN; 1 5; 2 NaN; NaN 1]);
assert(isequal(c, [1; 2; 1]));
assert(iscell(g) && isequal(size(g), [1 2]));
assert(isequal(g{1}(1:2), [1; 2]) && isnan(g{1}(3)));
assert(isequal(g{2}([1 3]), [5; 1]) && isnan(g{2}(2)));

% a row vector is several grouping variables with one row
[c, g] = groupcounts([3 1 3]);
assert(isequal(c, 1) && iscell(g) && isequal(size(g), [1 3]));

% cellstr column
[c, g] = groupcounts({'b'; ''; 'a'; ''});
assert(isequal(c, [2; 1; 1]));
assert(iscell(g) && isequal(size(g), [3 1]));
assert(isempty(g{1}) && strcmp(g{2}, 'a') && strcmp(g{3}, 'b'));

disp('SUCCESS');
