% Hexadecimal (0x..) and binary (0b..) integer literals (MATLAB R2019b+).
% numbl's number scanner had no 0x/0b handling, so `0x1F` lexed as Integer(0)
% followed by Ident("x1F") -- producing 0 plus an "undefined variable" error.

%!numbl:assert_jit c

assert(0x1F == 31, '0x1F should be 31');
assert(0xFF == 255, '0xFF should be 255');
assert(0x0 == 0, '0x0 should be 0');
assert(0b101 == 5, '0b101 should be 5');
assert(0b1111 == 15, '0b1111 should be 15');

% usable in expressions / indexing
v = [10 20 30 40];
assert(v(0x2) == 20, 'hex literal as index');
assert((0xA + 0b10) == 12, 'hex + binary arithmetic');

disp('SUCCESS');
