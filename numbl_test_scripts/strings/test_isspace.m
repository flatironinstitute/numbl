% isspace returns a logical array, true where the char is whitespace.
assert(isequal(isspace(sprintf('a b\tc\nd')), logical([0 1 0 1 0 1 0])), ...
    'mixed whitespace');
assert(islogical(isspace('x')), 'returns logical');
assert(isequal(size(isspace('abc')), [1 3]), 'same size as input');
assert(isequal(isspace('   '), logical([1 1 1])), 'all spaces');
disp('SUCCESS');
