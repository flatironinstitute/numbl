% 3D tensor: logical indexing and comparisons

a = reshape(1:12, 2, 3, 2);

% ── Comparison producing 3D logical/tensor ────────────────────────
b = a > 6;
assert(size(b, 1) == 2)
assert(size(b, 2) == 3)
assert(size(b, 3) == 2)
assert(b(1, 1, 1) == 0)
assert(b(2, 3, 1) == 0)
assert(b(1, 1, 2) == 1)
assert(b(2, 3, 2) == 1)

% ── Logical indexing: extract elements where condition is true ────
c = a(a > 6);
assert(length(c) == 6)
% Elements > 6 in column-major order: 7,8,9,10,11,12
assert(c(1) == 7)
assert(c(6) == 12)

% ── Logical indexing with combined conditions ─────────────────────
d = a(a >= 3 & a <= 8);
assert(length(d) == 6)
assert(d(1) == 3)
assert(d(6) == 8)

% ── sum/any/all on 3D logical results ─────────────────────────────
assert(sum(a(:) > 6) == 6)
assert(any(a(:) > 11))
assert(~all(a(:) > 6))

% ── Element-wise & and | on tensors ───────────────────────────────
x = [1 0 1 0];
y = [1 1 0 0];
z_and = x & y;
assert(z_and(1) == 1)
assert(z_and(2) == 0)
assert(z_and(3) == 0)
assert(z_and(4) == 0)
z_or = x | y;
assert(z_or(1) == 1)
assert(z_or(2) == 1)
assert(z_or(3) == 1)
assert(z_or(4) == 0)

% ── Element-wise & and | on 3D comparison results ────────────────
mask1 = a > 2;
mask2 = a < 10;
combined = mask1 & mask2;
assert(size(combined, 1) == 2)
assert(size(combined, 2) == 3)
assert(size(combined, 3) == 2)
% a(1,1,1)=1: 1>2=0, 1<10=1, 0&1=0
assert(combined(1, 1, 1) == 0)
% a(2,1,1)=2: 2>2=0, 2<10=1, 0&1=0
assert(combined(2, 1, 1) == 0)
% a(1,2,1)=3: 3>2=1, 3<10=1, 1&1=1
assert(combined(1, 2, 1) == 1)
% a(2,3,2)=12: 12>2=1, 12<10=0, 1&0=0
assert(combined(2, 3, 2) == 0)

% ── NOT on logical tensors ────────────────────────────────────────
notb = ~b;
assert(notb(1, 1, 1) == 1)
assert(notb(1, 1, 2) == 0)

% ── Logical indexing on 2D (regression) ───────────────────────────
m = [10 20 30 40 50];
r = m(m > 25);
assert(length(r) == 3)
assert(r(1) == 30)
assert(r(2) == 40)
assert(r(3) == 50)

% ── Logical indexing respects column-major order ──────────────────
% p = [1 2; 3 4; 5 6] stored column-major: 1,3,5,2,4,6
p = [1 2; 3 4; 5 6];
q = p(p > 3);
assert(length(q) == 3)
assert(q(1) == 5)
assert(q(2) == 4)
assert(q(3) == 6)

% ── Logical indexing with all true / all false ────────────────────
all_true = a(a > 0);
assert(length(all_true) == 12)
all_false = a(a > 100);
assert(length(all_false) == 0)

% ── Scalar comparison with logical index ──────────────────────────
v = [5 10 15 20];
w = v(v == 10);
assert(length(w) == 1)
assert(w == 10)

% ── find should still work as before ──────────────────────────────
idx = find(a > 6);
e = a(idx);
assert(length(e) == 6)
assert(e(1) == 7)
assert(e(6) == 12)

disp('SUCCESS')
