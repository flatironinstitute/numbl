% Test bitwise operations: bitand, bitor, bitxor, bitshift

% bitand
assert(bitand(5, 3) == 1);     % 101 & 011 = 001
assert(bitand(12, 10) == 8);   % 1100 & 1010 = 1000
assert(bitand(255, 15) == 15); % 11111111 & 00001111 = 00001111
assert(bitand(0, 255) == 0);

% bitor
assert(bitor(5, 3) == 7);      % 101 | 011 = 111
assert(bitor(12, 10) == 14);   % 1100 | 1010 = 1110
assert(bitor(0, 0) == 0);

% bitxor
assert(bitxor(5, 3) == 6);     % 101 ^ 011 = 110
assert(bitxor(12, 10) == 6);   % 1100 ^ 1010 = 0110
assert(bitxor(7, 7) == 0);

% bitshift left
assert(bitshift(1, 3) == 8);    % 1 << 3 = 8
assert(bitshift(5, 2) == 20);   % 101 << 2 = 10100

% bitshift right (negative shift)
assert(bitshift(8, -2) == 2);   % 1000 >> 2 = 10
assert(bitshift(16, -4) == 1);

% bitshift by 0
assert(bitshift(42, 0) == 42);

% Vectorized bitand
a = [5 12 255];
b = [3 10 15];
result = bitand(a, b);
assert(isequal(result, [1 8 15]));

% Vectorized bitor
result2 = bitor(a, b);
assert(isequal(result2, [7 14 255]));

% Vectorized bitxor
result3 = bitxor(a, b);
assert(isequal(result3, [6 6 240]));

disp('SUCCESS');
