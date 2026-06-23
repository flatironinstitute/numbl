% isreal across types: true only for real numerics, char, and logical.
assert(isreal(5), 'real scalar');
assert(~isreal(5i), 'complex scalar');
assert(isreal([1 2 3]), 'real vector');
assert(~isreal([1 2i 3]), 'complex vector');
assert(isreal('a'), 'char is real');
assert(isreal(true), 'logical is real');
assert(~isreal({1, 2}), 'cell is not real');
s.a = 1;
assert(~isreal(s), 'struct is not real');
assert(~isreal("str"), 'string is not real');
disp('SUCCESS');
