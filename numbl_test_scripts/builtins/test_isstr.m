% isstr is a legacy alias for ischar.

assert(isstr('hello'), 'char is str');
assert(isstr(['ab'; 'cd']), 'char matrix is str');
assert(~isstr(5), 'number is not str');
assert(~isstr(string('hi')), 'string-type is not char');
assert(~isstr({'a', 'b'}), 'cell is not str');
assert(isequal(isstr('x'), ischar('x')), 'isstr matches ischar');

disp('SUCCESS');
