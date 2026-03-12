% Test that logical indexing preserves vector orientation

% Row vector with logical index should stay row
r = [10 20 30 40];
mask = [true true false true];
result = r(mask);
assert(isequal(size(result), [1, 3]), 'logical indexing on row vector should preserve row orientation');
assert(isequal(result, [10 20 40]));

% Column vector with logical index should stay column
c = [10; 20; 30; 40];
mask_c = [true; true; false; true];
result_c = c(mask_c);
assert(isequal(size(result_c), [3, 1]), 'logical indexing on column vector should preserve column orientation');
assert(isequal(result_c, [10; 20; 40]));

% Row vector with isfinite logical mask
r2 = [1 Inf 3 4];
result2 = r2(isfinite(r2));
assert(isequal(size(result2), [1, 3]), 'isfinite logical indexing on row vector should preserve row');
assert(isequal(result2, [1 3 4]));

% Column vector with isfinite logical mask
c2 = [1; Inf; 3; 4];
result_c2 = c2(isfinite(c2));
assert(isequal(size(result_c2), [3, 1]), 'isfinite logical indexing on column vector should preserve column');
assert(isequal(result_c2, [1; 3; 4]));

disp('SUCCESS');
