% Logical-mask indexing of char arrays (read), and scalar-broadcast
% assignment into masked positions (char and numeric).

% --- read with a logical mask ---
w = 'abcd';
assert(strcmp(w(logical([1 0 1 0])), 'ac'), 'char logical read');
s = 'a,b,c';
assert(strcmp(s(s ~= ','), 'abc'), 'char mask from comparison');

% --- scalar broadcast assignment over multiple masked positions ---
t = sprintf('x\ny\nz');
t(t == sprintf('\n')) = ' ';
assert(strcmp(t, 'x y z'), 'char scalar-broadcast assign');

v = [1 2 3 4 5];
v(v > 3) = 0;
assert(isequal(v, [1 2 3 0 0]), 'numeric scalar-broadcast assign');

% --- count-matched (non-scalar) assignment still works ---
u = [10 20 30 40];
u(u > 20) = [7 8];
assert(isequal(u, [10 20 7 8]), 'count-matched assign');
disp('SUCCESS');
