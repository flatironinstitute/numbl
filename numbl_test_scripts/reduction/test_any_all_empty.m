% any/all on empty matrices must reduce along the first non-singleton
% dimension (like sum/prod), not collapse every empty to a scalar.

% 4x0: reduce along dim 1 (size 4) -> 1x0 empty
assert(isequal(size(any(false(4, 0))), [1 0]), 'any(4x0) size');
assert(isempty(any(false(4, 0))), 'any(4x0) empty');
assert(isequal(size(all(false(4, 0))), [1 0]), 'all(4x0) size');

% 0x4: reduce along dim 1 (size 0) -> 1x4 of identity (any=false, all=true)
assert(isequal(any(false(0, 4)), logical([0 0 0 0])), 'any(0x4)');
assert(isequal(all(false(0, 4)), logical([1 1 1 1])), 'all(0x4)');

% 0x0 and vectors still collapse to a scalar
assert(isequal(size(any([])), [1 1]), 'any([]) scalar');
assert(any([]) == false, 'any([]) value');
assert(all([]) == true, 'all([]) value');
assert(isequal(size(any(zeros(1, 0))), [1 1]), 'any(1x0) scalar');
assert(isequal(size(any(zeros(0, 1))), [1 1]), 'any(0x1) scalar');

% explicit dim and 'all' flag unaffected
assert(isequal(any(false(0, 4), 1), logical([0 0 0 0])), 'any(...,1)');
assert(any(false(4, 0), 'all') == false, 'any(...,all)');

disp('SUCCESS');
