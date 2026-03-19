% Edge case tests for reduction operations.

% --- sort: all-NaN descending should preserve original order ---
% Bug: numbl reverses NaN indices in descending sort (returns [3 2 1] instead of [1 2 3])
x = [NaN NaN NaN];
[y, idx] = sort(x, 'descend');
assert(all(isnan(y)), 'all-NaN descend should be all NaN');
assert(isequal(idx, [1 2 3]), 'all-NaN descend indices should preserve original order');

[y2, idx2] = sort(x, 'ascend');
assert(all(isnan(y2)), 'all-NaN ascend should be all NaN');
assert(isequal(idx2, [1 2 3]), 'all-NaN ascend indices should preserve original order');

% --- sort: mixed NaN descend stability ---
x3 = [NaN 3 NaN 1];
[y3, idx3] = sort(x3, 'descend');
assert(isnan(y3(1)) && isnan(y3(2)), 'NaN should be at beginning in descend');
assert(isequal(idx3(1:2), [1 3]), 'NaN indices should preserve relative order');
assert(isequal(y3(3:4), [3 1]), 'non-NaN values should be sorted descend');

% --- sort: empty input ---
[ye, idxe] = sort([]);
assert(isempty(ye), 'sort empty should return empty');
assert(isempty(idxe), 'sort empty idx should return empty');

% --- unique: NaN should be treated as distinct ---
% Bug: numbl deduplicates NaN because NaN.toString() == "NaN" for all NaN values
[c3] = unique([1 NaN 2 NaN 3]);
assert(length(c3) == 5, 'unique should treat each NaN as distinct');
assert(sum(isnan(c3)) == 2, 'unique should keep both NaN values');

% --- unique: NaN outputs with ia/ic ---
[C, ia, ic] = unique([1 NaN 2 NaN 3]);
assert(length(C) == 5, 'unique C should have 5 elements');
assert(length(ia) == 5, 'unique ia should have 5 elements');
assert(isequal(ia', [1 3 5 2 4]), 'unique ia should index all elements');

% --- min/max with all-NaN ---
x = [NaN NaN];
assert(isnan(min(x)), 'min of all NaN should be NaN');
assert(isnan(max(x)), 'max of all NaN should be NaN');

% --- min/max with NaN and numbers (2-arg element-wise) ---
assert(min(3, NaN) == 3, 'min(3,NaN) should be 3');
assert(min(NaN, 3) == 3, 'min(NaN,3) should be 3');
assert(max(3, NaN) == 3, 'max(3,NaN) should be 3');
assert(max(NaN, 3) == 3, 'max(NaN,3) should be 3');

% --- sum/prod of empty ---
assert(sum([]) == 0, 'sum of empty should be 0');
assert(prod([]) == 1, 'prod of empty should be 1');

% --- unique: orientation preserved ---
[c1] = unique([3 1 2]);
assert(isequal(size(c1), [1 3]), 'unique of row vec should be row vec');
[c2] = unique([3; 1; 2]);
assert(isequal(size(c2), [3 1]), 'unique of col vec should be col vec');

% --- cumsum along dim exceeding ndims ---
A = [1 2; 3 4];
r = cumsum(A, 3);
assert(isequal(r, A), 'cumsum along dim>ndims should return copy');

% --- diff of scalar and 1-element ---
assert(isempty(diff(5)), 'diff of scalar should be empty');
assert(isempty(diff([5])), 'diff of 1-element should be empty');

% --- find with direction='last' ---
x = [0 3 0 5 0];
idx = find(x, 1, 'last');
assert(idx == 4, 'find last should return last nonzero index');

% --- find with n > nnz ---
x = [0 1 0 1 0];
idx = find(x, 10);
assert(isequal(idx, [2 4]), 'find with n>nnz should return all');

% --- setdiff with NaN ---
r = setdiff([1 NaN 2 3], [2 3]);
assert(length(r) == 2, 'setdiff should keep NaN and 1');
assert(ismember(1, r), 'setdiff should keep 1');

% --- ismember with repeated values in B ---
[tf, loc] = ismember([3 1 2], [5 3 1 3]);
assert(isequal(tf, logical([1 1 0])), 'ismember tf wrong');
assert(loc(1) == 2, 'ismember should return first occurrence in B');
assert(loc(2) == 3, 'ismember loc for 1 wrong');
assert(loc(3) == 0, 'ismember loc for missing should be 0');

disp('SUCCESS');
