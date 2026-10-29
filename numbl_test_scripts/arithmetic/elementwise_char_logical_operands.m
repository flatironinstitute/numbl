% Char, logical and complex operands of element-wise operators.

% A multi-row char array is its character codes, in column-major order.
c = ['ab'; 'cd'];
assert(isequal(c + 0, [97 98; 99 100]));
assert(isequal(c < 99, logical([1 1; 0 0])));
assert(isequal(-c, -[97 98; 99 100]));
assert(isequal(c .^ 1, [97 98; 99 100]));
assert(isequal(c == 'b', logical([0 1; 0 0])));

% Char combines with complex values.
z = 'a' + 1i;
assert(real(z) == 97 && imag(z) == 1);
z = 'ab' .* (1 + 2i);
assert(isequal(z, [97 + 194i, 98 + 196i]));
assert(isequal('ab' == [97 98 + 1i], [true false]));

% &, | and xor expand singleton dimensions.
x = [1; 0; 2];
y = [1 0];
assert(isequal(x & y, logical([1 0; 0 0; 1 0])));
assert(isequal(x | y, logical([1 1; 1 0; 1 1])));
assert(isequal(xor(x, y), logical([0 1; 1 0; 0 1])));
assert(isequal(and(x, y), x & y));
assert(isequal(or(x, y), x | y));
assert(isequal(xor([1 0 1], [1; 0]), logical([0 1 0; 1 0 1])));
assert(islogical(x & y));

% Real zero times a complex scalar is +0.
r = 0 .* (-1.5 - 0.5i);
assert(isreal(r) && r == 0 && 1 / r == Inf);
r = (-1.5 - 0.5i) * 0;
assert(isreal(r) && 1 / r == Inf);
% Real times complex scales each part: a zero part stays 0 (no Inf*0).
r = Inf .* complex(0, 1);
assert(real(r) == 0 && imag(r) == Inf);

disp('SUCCESS')
