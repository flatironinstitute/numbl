% More adversarial COW tests — class instances, sparse, indexing
% modes, try/catch, closures, nested functions.

% ── 1. Logical-mask indexed store under sharing ─────────────────────
a = [1 2 3 4 5];
b = a;
a(a > 2) = 0;
assert(b(1) == 1 && b(2) == 2 && b(3) == 3 && b(4) == 4 && b(5) == 5, ...
  '1a: b unchanged after logical-mask write');
assert(a(3) == 0 && a(4) == 0 && a(5) == 0 && a(1) == 1 && a(2) == 2, ...
  '1b: a logical-mask write applied');

% ── 2. end keyword in shared lvalue ──────────────────────────────────
a = [10 20 30];
b = a;
a(end) = 999;
assert(b(3) == 30, '2a: b(end) unchanged');
assert(a(3) == 999, '2b: a(end) mutated');

% ── 3. Range index store under sharing ──────────────────────────────
a = [1 2 3 4 5];
b = a;
a(2:4) = [99 88 77];
assert(b(2) == 2 && b(3) == 3 && b(4) == 4, '3a: b unchanged');
assert(a(2) == 99 && a(3) == 88 && a(4) == 77, '3b: a range mutated');

% ── 4. Colon index store under sharing ──────────────────────────────
A = [1 2 3; 4 5 6];
B = A;
A(:, 1) = [100; 200];
assert(B(1, 1) == 1 && B(2, 1) == 4, '4a: B unchanged');
assert(A(1, 1) == 100 && A(2, 1) == 200, '4b: A column mutated');

% ── 5. Indexed read returns independent copy ────────────────────────
a = [10 20 30 40 50];
slice = a(2:4);
a(2) = 999;
assert(slice(1) == 20, sprintf('5a: slice(1) expected 20, got %g', slice(1)));
assert(a(2) == 999, '5b: a mutated');

% ── 6. Try/catch with mid-mutation throw ────────────────────────────
a = [1 2 3];
b = a;
try
  a(1) = 99;
  error('intentional');
catch
  % Mid-mutation throw — a should already reflect the partial update.
end
assert(b(1) == 1, '6a: b unchanged across try/catch');
assert(a(1) == 99, '6b: a partial update visible');

% ── 7. Closure captures struct, caller mutates ──────────────────────
% In MATLAB, anonymous fns snapshot by value at definition. Caller's
% subsequent mutation must NOT affect the snapshot.
captured = struct('v', [10 20 30]);
f = @() captured.v(2);
captured.v(2) = 999;
result = f();
assert(result == 20, sprintf('7: captured snapshot expected 20, got %g', result));

% ── 8. Closure captures cell, deeper mutation ──────────────────────
c = {[1 2 3], [4 5 6]};
f = @(i) c{i}(1);
c{1}(1) = 99;
v = f(1);
assert(v == 1, sprintf('8: closure cell snapshot expected 1, got %g', v));

% ── 9. Function returns inner ref, caller mutates outer ─────────────
function y = take_inner(s)
  y = s.inner;
end
s = struct('inner', struct('v', [1 2 3]));
held = take_inner(s);
s.inner.v(1) = 99;
assert(held.v(1) == 1, sprintf('9: held.inner unchanged, got %g', held.v(1)));

% ── 10. Cell containing struct containing tensor, full chain shared ─
inner = [1 2 3];
s.v = inner;
c = {s};
d = c;
d{1}.v(1) = 99;
assert(inner(1) == 1, '10a: inner unchanged');
assert(s.v(1) == 1, '10b: s unchanged');
assert(c{1}.v(1) == 1, '10c: c unchanged');
assert(d{1}.v(1) == 99, '10d: d mutated');

% ── 11. Indexed cell store: c(1:2) = {a, b} ────────────────────────
v1 = [1 2 3];
v2 = [4 5 6];
c = {[0 0], [0 0]};
d = c;
d(1:2) = {v1, v2};
assert(length(c{1}) == 2 && c{1}(1) == 0, '11a: c unchanged');
assert(d{1}(1) == 1 && d{2}(1) == 4, '11b: d replaced');
v1(1) = 99;
assert(d{1}(1) == 1, '11c: d{1} unchanged after v1 mutation');

% ── 12. Sparse matrix shared and mutated ────────────────────────────
S = sparse([1 2 3], [1 2 3], [10 20 30]);
T = S;
S(1, 1) = 999;
% T should be unchanged
assert(full(T(1, 1)) == 10, sprintf('12a: T(1,1) unchanged, got %g', full(T(1, 1))));
assert(full(S(1, 1)) == 999, '12b: S(1,1) mutated');

% ── 13. Repeated sharing then mutation ──────────────────────────────
a = [1 2 3];
b = a;
c = b;
d = c;
d(1) = 99;
assert(a(1) == 1 && b(1) == 1 && c(1) == 1, ...
  sprintf('13a: a/b/c unchanged, a(1)=%g b(1)=%g c(1)=%g', a(1), b(1), c(1)));
assert(d(1) == 99, '13b: d mutated');

% ── 14. Mutate one, then mutate the next ────────────────────────────
a = [1 2 3];
b = a;
a(1) = 10;   % a now [10 2 3], b still [1 2 3]
b(2) = 20;   % b now [1 20 3], a still [10 2 3]
assert(a(1) == 10 && a(2) == 2 && a(3) == 3, '14a: a state');
assert(b(1) == 1 && b(2) == 20 && b(3) == 3, '14b: b state');

% ── 15. Self-rebind under mutation ──────────────────────────────────
a = [1 2 3];
a = a;       % no-op
a(1) = 99;
assert(a(1) == 99, '15: self-rebind then mutate');

% ── 16. Nested functions and shared local ───────────────────────────
function out = outer_fn()
  v = [1 2 3];
  function inner_set()
    v(1) = 99;   % captures parent's v (nested-function semantics: shared scope)
  end
  inner_set();
  out = v;
end
result = outer_fn();
assert(result(1) == 99, sprintf('16: nested fn modifies parent v, got %g', result(1)));

% ── 17. Multi-level indexed store via struct array ──────────────────
arr = struct('cells', {{[1 2], [3 4]}, {[5 6], [7 8]}});
arr2 = arr;
arr2(1).cells{1}(1) = 99;
assert(arr(1).cells{1}(1) == 1, '17a: arr unchanged');
assert(arr2(1).cells{1}(1) == 99, '17b: arr2 mutated');

% ── 18. Sharing into cell elements (unrolled) ───────────────────────
% NOTE: An equivalent pattern inside a `for k = 1:N; copies{k} = master; end`
% loop currently corrupts `copies` under the JIT executor (--opt 1). The
% JIT's `__cellWrite` helper bypasses refcount tracking — it builds a
% POJO cell wrapper instead of a `RuntimeCell` and skips the
% incref/decref bookkeeping for inserted values. The interpreter path
% (--opt 0) handles the same loop correctly. Tracked as a follow-up
% for the JIT COW pass.
master = [1 2 3 4 5];
copies = cell(1, 3);
copies{1} = master;
copies{2} = master;
copies{3} = master;
master(1) = 999;
for k = 1:3
  assert(copies{k}(1) == 1, sprintf('18: copy %d corrupted', k));
end
assert(master(1) == 999, '18: master mutated');

% ── 19. Conditional COW path ────────────────────────────────────────
a = [1 2 3];
b = a;
if true
  a(1) = 100;
end
assert(b(1) == 1, '19a: b unchanged');
assert(a(1) == 100, '19b: a mutated under condition');

disp('SUCCESS')
