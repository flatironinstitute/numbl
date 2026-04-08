% Test that A(i,j) on a tensor with rank > 2 collapses trailing dimensions
% into the last index (MATLAB rule: unused trailing dims are linearized
% into whichever index position is the last one supplied).

% Base case: 3D tensor, A(:,:) — collapses dims 2 and 3 into one
B = reshape(1:24, 2, 3, 4);
y = B(:, :);
assert(isequal(size(y), [2 12]), sprintf('expected [2 12], got %s', mat2str(size(y))));
% Column-major layout: B(r, c, k) ↔ y(r, (k-1)*3 + c)
assert(y(1, 5) == 9, sprintf('y(1,5) = %g (expected 9)', y(1, 5)));
assert(y(2, 12) == 24, sprintf('y(2,12) = %g (expected 24)', y(2, 12)));

% Scalar row index + colon on remaining
z = B(1, :);
assert(isequal(size(z), [1 12]), sprintf('row extract size %s', mat2str(size(z))));
assert(z(1) == 1, 'z(1)');
assert(z(12) == 23, 'z(12)');

% Scalar row + scalar collapsed-col index
assert(B(2, 6) == 12, sprintf('B(2,6) = %g (expected 12)', B(2, 6)));
assert(B(1, 10) == 19, sprintf('B(1,10) = %g (expected 19)', B(1, 10)));

% 4D tensor, A(i,j) — trailing 2 dims collapse
C = reshape(1:48, 2, 3, 4, 2);
w = C(:, :);
assert(isequal(size(w), [2 24]), sprintf('4D collapse size %s', mat2str(size(w))));
assert(w(1, 1) == 1, 'w(1,1)');
assert(w(2, 24) == 48, 'w(2,24)');

% Linear indexing still works on N-D
assert(B(17) == 17, 'linear index');
assert(C(48) == 48, 'linear index 4D');

disp('SUCCESS');
