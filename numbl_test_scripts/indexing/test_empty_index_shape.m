% Test: indexing with empty arrays should produce correct output shapes
% MATLAB rules:
%   x([]) where [] is 0x0 -> always 0x0 regardless of base shape
%   x(ones(1,0)) where idx is 1x0 -> preserves base orientation
%   x(ones(0,1)) where idx is 0x1 -> preserves base orientation

r = [1 2 3];
c = [1; 2; 3];
s = 5;

% Index with 0x0 empty ([])
idx_sq = [];
assert(isequal(size(r(idx_sq)), [0 0]), 'row([]) should be [0 0]');
assert(isequal(size(c(idx_sq)), [0 0]), 'col([]) should be [0 0]');
assert(isequal(size(s(idx_sq)), [0 0]), 'scalar([]) should be [0 0]');

% Index with 1x0 empty
idx_row = ones(1,0);
assert(isequal(size(r(idx_row)), [1 0]), 'row(1x0) should be [1 0]');
assert(isequal(size(c(idx_row)), [0 1]), 'col(1x0) should be [0 1]');
assert(isequal(size(s(idx_row)), [1 0]), 'scalar(1x0) should be [1 0]');

% Index with 0x1 empty
idx_col = ones(0,1);
assert(isequal(size(r(idx_col)), [1 0]), 'row(0x1) should be [1 0]');
assert(isequal(size(c(idx_col)), [0 1]), 'col(0x1) should be [0 1]');
assert(isequal(size(s(idx_col)), [0 1]), 'scalar(0x1) should be [0 1]');

% Ensure 1x1 tensor (from matrix operation) also works
M = [1 2; 3 4];
t = M(1,1) + M(2,2);  % likely stored as 1x1 tensor internally
assert(isequal(size(t(idx_sq)), [0 0]), '1x1_tensor([]) should be [0 0]');

disp('SUCCESS');
