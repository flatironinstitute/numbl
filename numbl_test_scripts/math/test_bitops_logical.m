% Bitwise ops accept logical operands (treated as 0/1), like MATLAB.
assert(bitand(1, true) == 1, 'bitand 1 & true');
assert(bitand(3, true) == 1, 'bitand 3 & true');
assert(bitor(0, true) == 1, 'bitor 0 | true');
assert(bitxor(1, true) == 0, 'bitxor 1 ^ true');
assert(bitand(true, true) == 1, 'both logical');
assert(isequal(bitand([1 2 3], true), [1 0 1]), 'array & logical scalar');
disp('SUCCESS');
