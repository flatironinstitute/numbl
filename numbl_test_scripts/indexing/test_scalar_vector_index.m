% Indexing a scalar by a vector of ones must replicate the scalar for
% each index.  In MATLAB, every numeric value is a 1x1 matrix, so
%   r = 5; r([1 1 1])
% returns a 1x3 row vector [5 5 5].  This is relied on by
% chunkie/chunkerkerneval which does
%   rowdims(ids(:))
% where rowdims can be a 1x1 (when mk == 1) and ids is a long vector of
% ones.

% --- scalar by row vector ---
r = 5;
y1 = r([1 1 1]);
assert(isequal(size(y1), [1 3]), 'y1 shape');
assert(isequal(y1, [5 5 5]), 'y1 values');

% --- scalar by column vector ---
y2 = r([1; 1; 1]);
assert(isequal(size(y2), [3 1]), 'y2 shape');
assert(isequal(y2, [5; 5; 5]), 'y2 values');

% --- scalar by 2-D index matrix ---
y3 = r(ones(2, 3));
assert(isequal(size(y3), [2 3]), 'y3 shape');
assert(all(y3(:) == 5), 'y3 values');

% --- scalar by 1x1 index (trivial case, must still work) ---
y4 = r(1);
assert(isequal(y4, 5), 'y4 value');

% --- scalar by empty index ---
y5 = r([]);
assert(isempty(y5), 'y5 empty');

% --- chunkie pattern: scalar used as lookup table via a ones-vector ---
rowdims = 5;                % what numbl produces for a 1x1 degenerate case
ids = ones(8, 1);
result = rowdims(ids(:));
assert(isequal(size(result), [8 1]), 'chunkie result shape');
assert(all(result == 5), 'chunkie result values');

% --- and used in an LHS slice assignment, which is the actual chunkie use ---
ntarg = 8;
itargstart = zeros(ntarg + 1, 1);
itargstart(2:end) = rowdims(ids(:));
itargstart = 1 + cumsum(itargstart);
assert(isequal(itargstart, (1:5:5*8+1)'), 'cumsum pattern');

% --- negative: out-of-bounds index must still error ---
caught = false;
try
    bad = r([1 2]);
    disp(bad);
catch
    caught = true;
end
assert(caught, 'out-of-bounds should error');

disp('SUCCESS');
