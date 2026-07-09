% Test: linear-indexing a 1x1 tensor with a vector takes the INDEX's shape.
% MATLAB treats a 1x1 array as a scalar, so the "result follows the base's
% orientation" rule for vector bases must not kick in (Tony's trick:
% x(ones(m,1)) is m-by-1). Regression: 1x1 tensors from builtins like
% linspace/ones were treated as row vectors, so x([1;1]) came back 1x2.
% This broke surfacefun's trianglepts (via recursive barycentric weights).

x = linspace(0, 1, 1).';   % 1x1 tensor
assert(isequal(size(x([1;1])), [2 1]), 'tensor 1x1 with col idx should be [2 1]');
assert(isequal(size(x([1 1])), [1 2]), 'tensor 1x1 with row idx should be [1 2]');
assert(isequal(size(x(ones(3,1))), [3 1]), 'Tony''s trick should give a column');
assert(isequal(size(x(ones(2,2))), [2 2]), 'matrix idx keeps its shape');
assert(isequal(size(x(ones(0,1))), [0 1]), 'tensor 1x1 with 0x1 idx should be [0 1]');
assert(isequal(size(x(ones(1,0))), [1 0]), 'tensor 1x1 with 1x0 idx should be [1 0]');

y = ones(1, 1);            % another 1x1 tensor producer
assert(isequal(size(y([1;1;1])), [3 1]), 'ones(1,1) with col idx should be [3 1]');

% Non-scalar vectors still follow the base's orientation
r = [1 2 3];
c = [1; 2; 3];
assert(isequal(size(r([1;2])), [1 2]), 'row base keeps row orientation');
assert(isequal(size(c([1 2])), [2 1]), 'col base keeps col orientation');

% The surfacefun idiom that surfaced the bug: gather then add to a column
b = zeros(3, 1);
br = x([1;1]);
b(2:3) = b(2:3) + 2 * br(1:2);
assert(isequal(b, [0; 2; 2]), 'column accumulate after 1x1 gather');

disp('SUCCESS');
