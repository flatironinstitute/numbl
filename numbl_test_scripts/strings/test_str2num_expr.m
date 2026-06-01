% str2num evaluates its argument as a MATLAB expression, so it handles vectors,
% bracketed matrices and arithmetic -- not just a single scalar. numbl
% implemented it as Number(s), silently returning [] for anything non-scalar.

assert(isequal(str2num('1 2 3'), [1 2 3]), 'str2num space-separated row');
assert(isequal(str2num('[1 2 3]'), [1 2 3]), 'str2num bracketed');
assert(isequal(str2num('[1 2; 3 4]'), [1 2; 3 4]), 'str2num matrix');
assert(isequal(str2num('2+3'), 5), 'str2num arithmetic expression');
assert(isequal(str2num('42'), 42), 'str2num scalar still works');

disp('SUCCESS');
