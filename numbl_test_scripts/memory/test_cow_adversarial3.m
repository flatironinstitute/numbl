% Third wave of adversarial COW tests — exotic edge cases.

% ── 1. Linear (:) assignment under sharing ──────────────────────────
A = [1 2; 3 4];
B = A;
A(:) = 99;
assert(B(1, 1) == 1 && B(2, 2) == 4, '1a: B unchanged');
assert(A(1, 1) == 99 && A(2, 2) == 99, '1b: A all-99');

% ── 2. Range delete on shared ───────────────────────────────────────
a = [1 2 3 4 5];
b = a;
a(2:4) = [];
assert(length(b) == 5 && b(3) == 3, '2a: b length unchanged');
assert(length(a) == 2 && a(1) == 1 && a(2) == 5, '2b: a deleted middle');

% ── 3. Cell delete on shared ────────────────────────────────────────
c = {[1 2], [3 4], [5 6]};
d = c;
c(2) = [];
assert(length(d) == 3, '3a: d length unchanged');
assert(length(c) == 2, '3b: c shorter');
assert(d{2}(1) == 3 && c{2}(1) == 5, '3c: contents differ');

% ── 4. Self-rebind via index store at same position ─────────────────
a = [10 20 30];
b = a;
a(1) = a(1);   % rc=2 still triggers COW (value semantics)
assert(b(1) == 10, '4a: b unchanged');
assert(a(1) == 10, '4b: a unchanged');

% ── 5. Matrix slicing returns independent copy ──────────────────────
A = [1 2 3; 4 5 6; 7 8 9];
sub = A(:, 2);     % column 2
A(2, 2) = 999;
assert(sub(2) == 5, sprintf('5: sub unchanged after A mutation, got %g', sub(2)));

% ── 6. Function that does y = x; y(1) = 99; without further return ──
function modify_local_copy(x)
  y = x;
  y(1) = 99;
  % Should not affect caller's x — y is local.
end
v = [1 2 3];
modify_local_copy(v);
assert(v(1) == 1, '6: caller v unchanged after callee local modification');

% ── 7. Closure captures struct, mutate field of capture ─────────────
captured = struct('a', [1 2 3]);
f = @() captured.a(2);
captured.a(2) = 99;
v = f();
assert(v == 2, sprintf('7: closure captured snapshot, got %g', v));

% ── 8. Struct with multiple alias paths to same tensor ──────────────
v = [1 2 3];
s.left = v;
s.right.inner = v;
% Three paths: v, s.left, s.right.inner. All share T.
% Mutate via s.left:
s.left(1) = 99;
assert(v(1) == 1, '8a: v unchanged');
assert(s.left(1) == 99, '8b: s.left mutated');
assert(s.right.inner(1) == 1, '8c: s.right.inner unchanged');

% ── 9. Cell of cells, all sharing one tensor ────────────────────────
v = [1 2 3];
inner = {v};
c = {inner, inner};
c{1}{1}(1) = 99;
assert(v(1) == 1, '9a: v unchanged');
assert(inner{1}(1) == 1, '9b: inner unchanged');
assert(c{2}{1}(1) == 1, '9c: c{2} unchanged');
assert(c{1}{1}(1) == 99, '9d: c{1} mutated');

% ── 10. Mutation through a struct array element ─────────────────────
arr = struct('x', {[1 2 3], [4 5 6]});
v = arr(1).x;     % alias element 1's x
arr(1).x(1) = 99;
assert(v(1) == 1, '10a: v unchanged');
assert(arr(1).x(1) == 99, '10b: arr(1).x mutated');
assert(arr(2).x(1) == 4, '10c: arr(2) untouched');

% ── 11. Aliased then iterate-then-mutate ────────────────────────────
a = [1 2 3 4 5];
b = a;
total = 0;
for k = 1:length(a)
  total = total + a(k);
end
a(1) = 999;
assert(b(1) == 1, '11a: b unchanged across iteration + mutate');
assert(total == 15, '11b: sum correct');

% ── 12. Matrix transpose alias ──────────────────────────────────────
A = [1 2; 3 4];
B = A';
A(1, 1) = 999;
assert(B(1, 1) == 1, sprintf('12: transpose copy, got %g', B(1, 1)));

% ── 13. Concatenation with self ─────────────────────────────────────
a = [1 2 3];
b = [a, a];     % b is [1 2 3 1 2 3]
a(1) = 99;
assert(b(1) == 1 && b(4) == 1, '13: b independent of a after concat');
assert(a(1) == 99, '13: a mutated');

% ── 14. Ternary-like (if branch returns aliased) ────────────────────
function out = pick(cond, x, y)
  if cond
    out = x;
  else
    out = y;
  end
end
v = [1 2 3];
w = pick(true, v, [9 9 9]);
v(1) = 99;
assert(w(1) == 1, sprintf('14: w unchanged, got %g', w(1)));

% ── 15. Deeply nested chain mutate via different path ───────────────
s.a.b.c = [1 2 3];
t = s;
t.a.b.c(1) = 99;
assert(s.a.b.c(1) == 1, '15a: s deep chain unchanged');
assert(t.a.b.c(1) == 99, '15b: t deep chain mutated');
% Now mutate t.a.b.c again — already isolated chain.
t.a.b.c(2) = 88;
assert(t.a.b.c(1) == 99 && t.a.b.c(2) == 88, '15c: repeated mutate');
assert(s.a.b.c(2) == 2, '15d: s still unchanged');

% ── 16. Cell with mixed types, one element shared ───────────────────
v = [1 2 3];
c = {v, 'hello', struct('z', 42)};
d = c;
d{1}(2) = 99;
assert(v(2) == 2, '16a: v unchanged');
assert(c{1}(2) == 2, '16b: c unchanged');
assert(d{1}(2) == 99, '16c: d mutated');

% ── 17. Multi-level cell share + chain mutation ─────────────────────
inner = [1 2 3 4];
c1 = {inner};
c2 = {c1};
c3 = c2;
c3{1}{1}(2) = 99;
assert(inner(2) == 2, '17a: inner unchanged');
assert(c1{1}(2) == 2, '17b: c1 unchanged');
assert(c2{1}{1}(2) == 2, '17c: c2 unchanged');
assert(c3{1}{1}(2) == 99, '17d: c3 mutated');

% ── 18. Struct passed to function returning a field ─────────────────
function field_copy = take_field(s)
  field_copy = s.x;
end
s.x = [10 20 30];
held = take_field(s);
s.x(1) = 999;
assert(held(1) == 10, sprintf('18: held(1) expected 10, got %g', held(1)));

% ── 19. Recursive function that takes-and-returns a tensor ──────────
function y = recurse_inc(x, n)
  if n <= 0
    y = x;
  else
    y = recurse_inc(x, n - 1);
  end
end
v = [1 2 3];
w = recurse_inc(v, 3);
v(1) = 99;
assert(w(1) == 1, sprintf('19: w unchanged after recursion, got %g', w(1)));

% ── 20. Two-level chained alias write ───────────────────────────────
% s and t share a struct; t.a and s.a share a tensor.
% Mutating t.a(1) should isolate t fully.
s.a = [10 20 30];
t = s;
also = t.a;     % adds another alias
t.a(1) = 99;
assert(s.a(1) == 10, '20a: s unchanged');
assert(also(1) == 10, '20b: also unchanged');
assert(t.a(1) == 99, '20c: t mutated');

disp('SUCCESS')
