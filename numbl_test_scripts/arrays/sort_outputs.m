% Sort with one and two output arguments

% ── Single output: sorted values ───────────────────────────────────
v = [3 1 4 1 5 9 2 6];
s = sort(v);
assert(s(1) == 1)
assert(s(2) == 1)
assert(s(3) == 2)
assert(s(8) == 9)

% ── Two outputs: sorted values and indices ─────────────────────────
[sv, si] = sort([3 1 4 1 5]);
assert(sv(1) == 1)
assert(sv(2) == 1)
assert(sv(3) == 3)
assert(sv(4) == 4)
assert(sv(5) == 5)
assert(si(1) == 2)
assert(si(2) == 4)
assert(si(3) == 1)
assert(si(4) == 3)
assert(si(5) == 5)

% ── Descending sort with indices ───────────────────────────────────
[sd, id] = sort([3 1 4 1 5], 'descend');
assert(sd(1) == 5)
assert(sd(5) == 1)
assert(id(1) == 5)

% ── Sort matrix columns with indices ──────────────────────────────
M = [3 6; 1 4; 2 5];
[sm, im] = sort(M);
assert(sm(1,1) == 1)
assert(sm(2,1) == 2)
assert(sm(3,1) == 3)
assert(im(1,1) == 2)
assert(im(2,1) == 3)
assert(im(3,1) == 1)
assert(sm(1,2) == 4)
assert(im(1,2) == 2)

% ── Sort with two outputs on max/min ──────────────────────────────
[mx, ix] = max([3 1 4 1 5]);
assert(mx == 5)
assert(ix == 5)

[mn, in2] = min([3 1 4 1 5]);
assert(mn == 1)
assert(in2 == 2)

disp('SUCCESS')
