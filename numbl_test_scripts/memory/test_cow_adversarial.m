% Adversarial tests for refcount-driven COW. Each block tries a
% pattern that could plausibly break the system; assertions verify the
% expected by-value semantics.

% ── 1. Struct-array element field mutation, array shared ─────────────
arr = struct('x', {[1 2 3], [4 5 6], [7 8 9]});
arr2 = arr;
arr2(2).x(1) = 999;
assert(arr(2).x(1) == 4, sprintf('1a: arr unchanged, got %g', arr(2).x(1)));
assert(arr2(2).x(1) == 999, '1b: arr2 mutated');
assert(arr(1).x(1) == 1 && arr2(1).x(1) == 1, '1c: untouched siblings agree');

% ── 2. Cell of structs, cell shared ─────────────────────────────────
c = {struct('v', [1 2 3]), struct('v', [4 5 6])};
d = c;
d{1}.v(2) = 999;
assert(c{1}.v(2) == 2, sprintf('2a: c{1}.v(2) expected 2, got %g', c{1}.v(2)));
assert(d{1}.v(2) == 999, '2b: d{1}.v(2) mutated');
assert(c{2}.v(1) == 4 && d{2}.v(1) == 4, '2c: untouched cell unchanged');

% ── 3. Nested cell, deep mutation ────────────────────────────────────
inner = [10 20 30];
c = {{inner, [40 50 60]}, [70 80 90]};
d = c;
d{1}{1}(2) = -1;
assert(inner(2) == 20, sprintf('3a: inner unchanged, got %g', inner(2)));
assert(c{1}{1}(2) == 20, '3b: c unchanged');
assert(d{1}{1}(2) == -1, '3c: d mutated');

% ── 4. Same tensor stored in two struct fields ───────────────────────
v = [1 2 3];
s.a = v;
s.b = v;
s.a(1) = 99;
assert(v(1) == 1, sprintf('4a: v unchanged, got %g', v(1)));
assert(s.a(1) == 99, '4b: s.a mutated');
assert(s.b(1) == 1, '4c: s.b unchanged (shared field semantics)');

% ── 5. RHS aliases LHS scalar element ────────────────────────────────
a = [10 20 30];
b = a;
a(1) = a(2);   % a(1) becomes 20
assert(b(1) == 10, sprintf('5a: b(1) unchanged, got %g', b(1)));
assert(a(1) == 20 && a(2) == 20, '5b: a(1) is now 20');

% ── 6. RHS reads from same chain being written ───────────────────────
s.x = [1 2 3];
t = s;
t.x(1) = t.x(3);   % t.x(1) becomes 3
assert(s.x(1) == 1 && s.x(3) == 3, '6a: s untouched');
assert(t.x(1) == 3 && t.x(3) == 3, '6b: t mutated correctly');

% ── 7. Concatenation aliasing ────────────────────────────────────────
v = [1 2 3];
M = [v; v];
v(1) = 99;
assert(M(1, 1) == 1 && M(2, 1) == 1, '7a: M unchanged after v mutation');
assert(v(1) == 99, '7b: v mutated');

% ── 8. Cell paren vs brace assignment ────────────────────────────────
c = {[1 2 3], [4 5 6]};
d = c;
d(1) = {[9 9 9]};   % paren replaces with the {...} contents
assert(c{1}(1) == 1, '8a: c{1} unchanged after d paren-assign');
assert(d{1}(1) == 9, '8b: d{1} replaced');

% ── 9. Element deletion under sharing ────────────────────────────────
a = [10 20 30 40];
b = a;
a(2) = [];
assert(length(b) == 4 && b(2) == 20, '9a: b unchanged by deletion');
assert(length(a) == 3 && a(2) == 30, '9b: a element removed');

% ── 10. Auto-grow under sharing ──────────────────────────────────────
a = [1 2 3];
b = a;
a(10) = 999;
assert(length(b) == 3, '10a: b length unchanged');
assert(length(a) == 10 && a(10) == 999, '10b: a grew');
assert(b(3) == 3, '10c: b values unchanged');

% ── 11. Multi-output assigns same value to two slots ─────────────────
function [x, y] = same_pair()
  v = [1 2 3];
  x = v;
  y = v;
end
[p, q] = same_pair();
p(1) = 99;
assert(q(1) == 1, sprintf('11: q(1) unchanged, got %g', q(1)));
assert(p(1) == 99, '11: p(1) mutated');

% ── 12. Function param mutation does not affect caller ───────────────
function modify_param(t)
  t(1) = 999;
end
held = [1 2 3];
modify_param(held);
assert(held(1) == 1, sprintf('12: held(1) unchanged, got %g', held(1)));

% ── 13. Function returns alias of param + caller mutates ─────────────
function y = pass_through(x)
  y = x;
end
p = [10 20 30];
q = pass_through(p);
p(2) = 999;
assert(q(2) == 20, '13a: q unchanged after p mutation');
assert(p(2) == 999, '13b: p mutated');

% ── 14. Function returns alias of param + callee mutates after ───────
% (This pattern would corrupt q if the alias check inside the function
% missed the caller-side hold.)
function y = mutate_after_alias(x)
  y = x;
  y(1) = 99;
  y(2) = 100;
end
p = [1 2 3];
q = mutate_after_alias(p);
assert(p(1) == 1 && p(2) == 2, '14a: p unchanged');
assert(q(1) == 99 && q(2) == 100, '14b: q has mutations');

% ── 15. Persistent variable shared across calls ──────────────────────
function out = persistent_cow()
  persistent stored;
  if isempty(stored); stored = [1 2 3]; end;
  out = stored;
  stored(1) = stored(1) + 1;
end
a = persistent_cow();   % a = [1 2 3], stored becomes [2 2 3]
b = persistent_cow();   % b = [2 2 3], stored becomes [3 2 3]
c = persistent_cow();   % c = [3 2 3], stored becomes [4 2 3]
assert(a(1) == 1, sprintf('15a: a(1) expected 1, got %g', a(1)));
assert(b(1) == 2, sprintf('15b: b(1) expected 2, got %g', b(1)));
assert(c(1) == 3, sprintf('15c: c(1) expected 3, got %g', c(1)));

% ── 16. Global variable shared ───────────────────────────────────────
function bump_global()
  global G
  G(1) = G(1) + 1;
end
function out = read_global()
  global G
  out = G;
end
global G
G = [10 20 30];
held = read_global();
bump_global();
assert(held(1) == 10, sprintf('16a: held(1) expected 10, got %g', held(1)));
assert(G(1) == 11, '16b: G(1) bumped');

% ── 17. Compound update via indexed compound expression ──────────────
a = [10 20 30];
b = a;
a(1) = a(1) + a(2);  % a(1) becomes 30
assert(b(1) == 10, '17a: b unchanged');
assert(a(1) == 30, '17b: a(1) is 30');

% ── 18. Deeply nested chain with mixed shapes ────────────────────────
s.a.b{1}.c.d = [1 2 3];
t = s;
t.a.b{1}.c.d(2) = 999;
assert(s.a.b{1}.c.d(2) == 2, '18a: s deep unchanged');
assert(t.a.b{1}.c.d(2) == 999, '18b: t deep mutated');

% ── 19. Sibling chains share an inner object ─────────────────────────
inner_struct = struct('v', [1 2 3]);
s.left = inner_struct;
s.right = inner_struct;
s.left.v(1) = 99;
assert(inner_struct.v(1) == 1, '19a: original inner unchanged');
assert(s.left.v(1) == 99, '19b: left mutated');
assert(s.right.v(1) == 1, '19c: right unchanged');

% ── 20. Cell self-reference via element copy ─────────────────────────
c = {[1 2 3]};
d = c;
c{2} = c{1};  % c is now {[1 2 3], [1 2 3]} (or same ref)
c{1}(1) = 99;
assert(c{1}(1) == 99, '20a: c{1} mutated');
assert(c{2}(1) == 1, '20b: c{2} unchanged (separate after COW)');
assert(d{1}(1) == 1, '20c: d unchanged');

disp('SUCCESS')
