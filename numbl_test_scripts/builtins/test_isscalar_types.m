% isscalar is numel(x)==1 for ANY type, not just numeric arrays.

% Numeric
assert(isscalar(5), 'numeric scalar');
assert(~isscalar([1 2 3]), 'numeric vector');
assert(isscalar(true), 'logical scalar');

% Cell arrays (regression: a 1x1 cell is a scalar)
assert(isscalar({64}), '1x1 cell is scalar');
assert(~isscalar({1, 2}), '1x2 cell is not scalar');
assert(~isscalar({}), 'empty cell is not scalar');

% Char and string
assert(isscalar('a'), 'single char is scalar');
assert(~isscalar('ab'), 'multi-char is not scalar');
assert(isscalar("hello"), 'string scalar is scalar');

% Struct
s.a = 1;
assert(isscalar(s), 'scalar struct is scalar');
sa(1).a = 1; sa(2).a = 2;
assert(~isscalar(sa), 'struct array is not scalar');

% The cmocean dispatch pattern: c{isscalar(c)} when c is a 1x1 cell.
c = {64};
assert(any(isscalar(c)), 'isscalar(c) is truthy');
assert(c{isscalar(c)} == 64, 'logical-true cell index returns element');

disp('SUCCESS');
