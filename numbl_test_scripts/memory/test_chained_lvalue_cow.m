% Refcount-driven COW must propagate through chained lvalues:
% mutating a leaf reached through a shared container must not leak to
% the other holder of that container.

% ── 1. Struct field, leaf tensor ────────────────────────
s.x = [10 20 30];
t = s;
t.x(2) = 999;
assert(s.x(2) == 20, sprintf('s.x(2) expected 20, got %g', s.x(2)));
assert(t.x(2) == 999, 't.x(2) expected 999');

% ── 2. Nested struct ────────────────────────────────────
s = struct();
s.outer.inner = [1 2 3];
t = s;
t.outer.inner(1) = 77;
assert(s.outer.inner(1) == 1, 'nested: s unchanged');
assert(t.outer.inner(1) == 77, 'nested: t mutated');

% ── 3. Cell wrapper alias ───────────────────────────────
v = [100 200 300];
c = {v};
d = c;
d{1}(1) = -5;
assert(v(1) == 100, 'cell alias: v unchanged');
assert(c{1}(1) == 100, 'cell alias: c unchanged');
assert(d{1}(1) == -5, 'cell alias: d mutated');

% ── 4. Struct holding cell holding tensor ───────────────
s = struct();
s.c = {[1 2 3]};
t = s;
t.c{1}(2) = 999;
assert(s.c{1}(2) == 2, 'mixed: s unchanged');
assert(t.c{1}(2) == 999, 'mixed: t mutated');

% ── 5. Tensor shared via env binding AND struct field ───
inner = [1 2 3];
s = struct();
s.x = inner;
t = s;
t.x(1) = 99;
assert(inner(1) == 1, 'multi-share: inner unchanged');
assert(s.x(1) == 1, 'multi-share: s unchanged');
assert(t.x(1) == 99, 'multi-share: t mutated');

% ── 6. Repeated mutations preserve isolation ────────────
s = struct();
s.x = [1 2 3];
t = s;
t.x(1) = 10;
t.x(2) = 20;
t.x(3) = 30;
assert(s.x(1) == 1 && s.x(2) == 2 && s.x(3) == 3, 's unchanged across reps');
assert(t.x(1) == 10 && t.x(2) == 20 && t.x(3) == 30, 't fully mutated');

disp('SUCCESS')
