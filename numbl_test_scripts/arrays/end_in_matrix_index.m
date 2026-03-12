% Using 'end' keyword inside bracket indices like a([1 end], :)

a = [10 20 30 40 50; 60 70 80 90 100; 110 120 130 140 150];

% a([1 end], :) should select first and last rows
b = a([1 end], :);
assert(size(b, 1) == 2)
assert(size(b, 2) == 5)
assert(b(1, 1) == 10)
assert(b(1, 5) == 50)
assert(b(2, 1) == 110)
assert(b(2, 5) == 150)

% a(:, [1 end]) should select first and last columns
c = a(:, [1 end]);
assert(size(c, 1) == 3)
assert(size(c, 2) == 2)
assert(c(1, 1) == 10)
assert(c(1, 2) == 50)
assert(c(3, 1) == 110)
assert(c(3, 2) == 150)

% end in arithmetic expressions inside brackets
d = a([end-1 end], :);
assert(size(d, 1) == 2)
assert(d(1, 1) == 60)
assert(d(2, 1) == 110)

% end with range inside brackets: [1:end]
v = [100 200 300 400 500];
e = v([1:end]);
assert(length(e) == 5)
assert(e(1) == 100)
assert(e(5) == 500)

% end in a partial range inside brackets: [2:end]
f = v([2:end]);
assert(length(f) == 4)
assert(f(1) == 200)
assert(f(4) == 500)

% [end] as single-element index vector
g = v([end]);
assert(g == 500)

% [1 2 end] with more than two elements
h = a([1 2 end], :);
assert(size(h, 1) == 3)
assert(h(1, 1) == 10)
assert(h(2, 1) == 60)
assert(h(3, 1) == 110)

% Both dimensions using end in brackets
k = a([1 end], [1 end]);
assert(size(k, 1) == 2)
assert(size(k, 2) == 2)
assert(k(1, 1) == 10)
assert(k(1, 2) == 50)
assert(k(2, 1) == 110)
assert(k(2, 2) == 150)

% end in arithmetic inside range expressions: a(1:end-1)
m = v(1:end-1);
assert(length(m) == 4)
assert(m(1) == 100)
assert(m(4) == 400)

% end-2 in range
n = v(2:end-1);
assert(length(n) == 3)
assert(n(1) == 200)
assert(n(3) == 400)

% ── Edge cases ──────────────────────────────────────────────────────

% end*1 and end/1 (multiplication and division with end)
p = v(end*1);
assert(p == 500)

% end used in both start and end of range
q = v(end-3:end-1);
assert(length(q) == 3)
assert(q(1) == 200)
assert(q(3) == 400)

% end with addition
r = a(end-2+1, :);
assert(r(1) == 60)

% bare end as index (already works, but verify not broken)
assert(v(end) == 500)
assert(a(end, end) == 150)

% end in range a(end:end)
s = v(end:end);
assert(length(s) == 1)
assert(s(1) == 500)

% end in range with step: a(end:-1:1)
t = v(end:-1:1);
assert(length(t) == 5)
assert(t(1) == 500)
assert(t(5) == 100)

% end arithmetic with step range: a(end:-1:end-2)
u = v(end:-1:end-2);
assert(length(u) == 3)
assert(u(1) == 500)
assert(u(3) == 300)

% [end-2:end] range inside brackets
w = v([end-2:end]);
assert(length(w) == 3)
assert(w(1) == 300)
assert(w(3) == 500)

% Nested: use end result as index into another array
x = [10 20 30 40 50 60 70 80 90 100];
y = x([1 end]);
assert(y(1) == 10)
assert(y(2) == 100)

% 2D end in both dims with arithmetic
z = a([1 end-1], [2 end]);
assert(size(z, 1) == 2)
assert(size(z, 2) == 2)
assert(z(1, 1) == 20)
assert(z(1, 2) == 50)
assert(z(2, 1) == 70)
assert(z(2, 2) == 100)

% Verify end resolves per-dimension correctly
% a is 3x5, so end in row dim = 3, end in col dim = 5
assert(a(end, 1) == 110)
assert(a(1, end) == 50)

disp('SUCCESS')
