% Concatenation with empty arrays

% ── Horizontal concat: grow vector in loop ─────────────────────────
result = [];
for i = 1:5
    if mod(i, 2) == 1
        result = [result, i^2];
    end
end
assert(length(result) == 3)
assert(result(1) == 1)
assert(result(2) == 9)
assert(result(3) == 25)

% ── Vertical concat: build matrix row by row ──────────────────────
M = [];
for i = 1:3
    M = [M; (i*10 + (1:4))];
end
assert(size(M, 1) == 3)
assert(size(M, 2) == 4)
assert(M(1,1) == 11)
assert(M(2,3) == 23)
assert(M(3,4) == 34)

% ── Empty with scalar ─────────────────────────────────────────────
a = [[], 5];
assert(length(a) == 1)
assert(a(1) == 5)

b = [5, []];
assert(length(b) == 1)
assert(b(1) == 5)

% ── Empty with vector ─────────────────────────────────────────────
c = [[], [1 2 3]];
assert(length(c) == 3)
assert(c(2) == 2)

d = [[1 2 3], []];
assert(length(d) == 3)

% ── Vertical empty with vector ────────────────────────────────────
e = [];
e = [e; [1 2 3]];
assert(size(e, 1) == 1)
assert(size(e, 2) == 3)
e = [e; [4 5 6]];
assert(size(e, 1) == 2)
assert(e(2,1) == 4)

disp('SUCCESS')
