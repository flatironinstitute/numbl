% Default-dim reductions must reduce along the first dimension whose size != 1,
% even when that dimension has size 0. numbl's firstReduceDim used
% shape.filter(d => d > 1), treating a size-0 dim as singleton and wrongly
% collapsing the whole array to a scalar.

% 0x3: first non-singleton dim is dim 1 (size 0) -> reduce to 1x3
assert(isequal(sum(zeros(0, 3)), [0 0 0]), 'sum(zeros(0,3))');
assert(isequal(size(sum(zeros(0, 3))), [1 3]), 'sum(zeros(0,3)) size');
assert(isequal(prod(zeros(0, 3)), [1 1 1]), 'prod(zeros(0,3)) (empty product = 1)');

% 3x0: first non-singleton dim is dim 1 (size 3) -> reduce to 1x0
assert(isequal(size(sum(zeros(3, 0))), [1 0]), 'sum(zeros(3,0)) size should be 1x0');
assert(isempty(sum(zeros(3, 0))), 'sum(zeros(3,0)) should be empty');

% explicit dim still correct
assert(isequal(sum(zeros(0, 3), 1), [0 0 0]), 'sum(...,1) over 0x3');

disp('SUCCESS');
