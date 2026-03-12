% 3D tensor: end sentinel in various indexing contexts

a = reshape(1:24, 2, 3, 4);

% ── end as scalar index in each dimension ─────────────────────────
assert(a(end, 1, 1) == 2)
assert(a(1, end, 1) == 5)
assert(a(1, 1, end) == 19)
assert(a(end, end, end) == 24)

% ── end in range expression: 1:end ────────────────────────────────
b = a(1:end, 1, 1);
assert(length(b) == 2)
assert(b(1) == 1)
assert(b(2) == 2)

c = a(1, 1:end, 1);
assert(length(c) == 3)
assert(c(1) == 1)
assert(c(3) == 5)

d = a(1, 1, 1:end);
assert(length(d) == 4)
assert(d(1) == 1)
assert(d(4) == 19)

% ── end arithmetic: end-1 ─────────────────────────────────────────
assert(a(end-1, 1, 1) == 1)
assert(a(1, end-1, 1) == 3)
assert(a(1, 1, end-1) == 13)

% ── end in range with arithmetic ──────────────────────────────────
e = a(1, 1, 2:end);
assert(length(e) == 3)
assert(e(1) == 7)
assert(e(3) == 19)

f = a(1, 1, 1:end-1);
assert(length(f) == 3)
assert(f(1) == 1)
assert(f(3) == 13)

% ── end in linear indexing on 3D (single index) ───────────────────
assert(a(end) == 24)

% ── end in assignment context ─────────────────────────────────────
g = reshape(1:8, 2, 2, 2);
g(end, end, end) = 99;
assert(g(2, 2, 2) == 99)
assert(g(1, 1, 1) == 1)

% ── end in range assignment ───────────────────────────────────────
h = reshape(1:12, 2, 3, 2);
h(1, 1:end, 2) = [10 20 30];
assert(h(1, 1, 2) == 10)
assert(h(1, 2, 2) == 20)
assert(h(1, 3, 2) == 30)
assert(h(2, 1, 2) == 8)

disp('SUCCESS')
