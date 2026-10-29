% The single class: conversion, class queries, operators, indexing,
% concatenation and colon. Verified against MATLAB R2025b.

% Conversion and class queries
x = single([1 2 3]);
assert(strcmp(class(x), 'single'));
assert(strcmp(class(single(3)), 'single'));
assert(isa(x, 'single') && isa(x, 'float') && isa(x, 'numeric'));
assert(~isa(x, 'double') && ~isa(x, 'integer'));
assert(isfloat(x) && isnumeric(x) && isreal(x) && ~islogical(x));
assert(strcmp(class(single(true)), 'single'));
assert(strcmp(class(single('ab')), 'single') && isequal(single('ab'), [97 98]));
assert(isequal(size(single('')), [0 0]) && strcmp(class(single('')), 'single'));
assert(strcmp(class(single(1+2i)), 'single') && ~isreal(single(1+2i)));
assert(strcmp(class(double(x)), 'double') && isequal(double(x), [1 2 3]));
assert(strcmp(class(double(single(2))), 'double'));
assert(strcmp(class(single(x)), 'single'));
assert(isequal(size(single(zeros(2, 0))), [2 0]));

% Rounding to single precision
assert(double(single(pi)) == 3.1415927410125732);
assert(double(single(16777217)) == 16777216);
assert(double(single(1/3)) == 0.3333333432674408);
assert(double(single(1e300)) == Inf);

% Arithmetic: a single operand makes the result single, computed in single
assert(strcmp(class(x / 3), 'single'));
y = single(1) / single(3);
assert(strcmp(class(y), 'single') && double(y) == 0.3333333432674408);
assert(double(single(0.1) * 3) == 0.30000001192092896);
assert(strcmp(class(single(1) + 1), 'single'));
assert(strcmp(class(1 - single(1)), 'single'));
assert(strcmp(class(single(1) + true), 'single'));
assert(strcmp(class(-x), 'single') && isequal(-x, [-1 -2 -3]));
assert(strcmp(class(+x), 'single'));
assert(strcmp(class(single(2)^2), 'single') && strcmp(class(2^single(2)), 'single'));
assert(strcmp(class(x .^ 2), 'single') && strcmp(class(2 .\ x), 'single'));
z = single(2) * 1e300;
assert(strcmp(class(z), 'single') && z == Inf);
assert(double(single(1e10) * single(1e30)) == Inf);
m = single(magic(3));
assert(strcmp(class(m * m), 'single') && isequal(m * m, magic(3) * magic(3)));
assert(strcmp(class(single(1) * [1 2; 3 4]), 'single'));
assert(strcmp(class(single([1 2; 3 4]) \ [1; 2]), 'single'));
assert(strcmp(class([1 2; 3 4] / single([1 2; 3 4])), 'single'));
c = single(1 + 2i);
assert(strcmp(class(c * 2), 'single') && isequal(c * 2, 2 + 4i));
assert(strcmp(class(c'), 'single') && isequal(c', 1 - 2i));

% Transpose keeps the class
assert(strcmp(class(x'), 'single') && isequal(size(x'), [3 1]));
assert(strcmp(class(x.'), 'single'));
assert(strcmp(class(single([1+1i 2]).'), 'single'));

% Comparisons and logical operators give logical
assert(strcmp(class(single(1) > 1), 'logical'));
assert(strcmp(class(x == 2), 'logical') && isequal(x == 2, [false true false]));
assert(strcmp(class(single(1) & 1), 'logical'));
assert(strcmp(class(~x), 'logical'));
assert(single(0.5) == 0.5 && single(pi) ~= pi);

% Indexing keeps the class
assert(strcmp(class(x(2)), 'single') && x(2) == 2);
assert(strcmp(class(x([1 2])), 'single'));
assert(strcmp(class(m(2, :)), 'single') && strcmp(class(m(:)), 'single'));
assert(strcmp(class(m(end)), 'single'));
s = single(5);
assert(strcmp(class(s(1)), 'single'));

% Indexed assignment keeps the target's class
a = [1 2 3];
a(2) = single(5);
assert(strcmp(class(a), 'double') && isequal(a, [1 5 3]));
b = single([1 2 3]);
b(2) = pi;
assert(strcmp(class(b), 'single') && double(b(2)) == 3.1415927410125732);
b(5) = 1/3;
assert(strcmp(class(b), 'single') && double(b(5)) == 0.3333333432674408);
assert(isequal(b(4), 0));
b(end + 1) = 1/3;
assert(strcmp(class(b), 'single') && double(b(6)) == 0.3333333432674408);
l = true(1, 3);
l(1) = single(5);
assert(strcmp(class(l), 'logical'));
d = [1 2 3];
d(2) = single(1 + 2i);
assert(strcmp(class(d), 'double') && isequal(d, [1 1+2i 3]));
q = single([1 2 3]);
q(q > 1) = 0;
assert(strcmp(class(q), 'single') && isequal(q, [1 0 0]));
q(:) = 1/3;
assert(strcmp(class(q), 'single') && all(double(q) == 0.3333333432674408));
w = zeros(2);
w(:) = single([1 2 3 4]);
assert(strcmp(class(w), 'double'));
e = [];
e(2) = single(3);
assert(strcmp(class(e), 'double'));
ch = 'abc';
ch(2) = single(66);
assert(strcmp(class(ch), 'char') && strcmp(ch, 'aBc'));
r = single([1 2 3]);
r(2) = [];
assert(strcmp(class(r), 'single') && isequal(r, [1 3]));
t = single([1 2]);
t2 = t;
t2(1) = 7;
assert(isequal(t, [1 2]) && isequal(t2, [7 2]) && strcmp(class(t2), 'single'));

% Concatenation: single wins over double and logical, char wins over single
assert(strcmp(class([single(1) 2]), 'single'));
assert(strcmp(class([1; single(2)]), 'single'));
assert(strcmp(class([true single(1)]), 'single'));
assert(strcmp(class([single([]) 1]), 'single'));
assert(strcmp(class([single(zeros(1, 0)) 1]), 'single'));
assert(double([single(1) 0.1]) * [0; 1] == double(single(0.1)));
assert(strcmp(class([single(65) 'a']), 'char') && strcmp([single(65) 'a'], 'Aa'));
assert(strcmp(class(['a' single(66)]), 'char'));
assert(strcmp(class([single(1) {2}]), 'cell'));

% Colon
assert(strcmp(class(single(1):3), 'single') && isequal(single(1):3, [1 2 3]));
assert(strcmp(class(1:single(3)), 'single'));
assert(strcmp(class(single(1):0.5:2), 'single'));

% For loops iterate single scalars and columns
for k = single([1 2])
  assert(strcmp(class(k), 'single'));
end
for k = single(1):2
  assert(strcmp(class(k), 'single'));
end
for col = single([1 2; 3 4])
  assert(strcmp(class(col), 'single') && isequal(size(col), [2 1]));
end

% Conditions and containers
if single(1)
  ok = true;
else
  ok = false;
end
assert(ok);
st.v = single(1);
assert(strcmp(class(st.v), 'single'));
cc = {single(2)};
assert(strcmp(class(cc{1}), 'single'));
assert(isequal(single([1 2]), [1 2]) && isequal(single(2), 2));

disp('SUCCESS')
