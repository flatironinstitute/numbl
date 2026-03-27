% Test deconv preserves column vector orientation
% MATLAB: deconv of column vectors returns column vectors

b = [1;3;5;3];
a = [1;1];

% Single output
q = deconv(b, a);
assert(isequal(size(q), [3, 1]), 'deconv q: should be column');
assert(isequal(q, [1;2;3]), 'deconv q: wrong values');

% Two outputs
[q2, r2] = deconv(b, a);
assert(isequal(size(q2), [3, 1]), 'deconv q2: should be column');
assert(isequal(size(r2), [4, 1]), 'deconv r2: should be column');

% Row vector input should stay row
q3 = deconv([1 3 5 3], [1 1]);
assert(isequal(size(q3), [1, 3]), 'deconv row: should be row');

disp('SUCCESS');
