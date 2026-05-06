% Stress tests for memory-pool correctness under heavy buffer churn.
%
% Each test sets up a tensor whose data should remain intact across
% many subsequent allocations of the SAME SIZE. If the pool releases
% the held buffer prematurely (use-after-free) and a fresh allocation
% reuses it, the held tensor's contents would change — the asserts
% catch that.
%
% Sized to favor pool reuse: most allocations are 100-element vectors
% to maximize same-size churn. Vitest test-mode routes every alloc
% through the pool regardless of size.

% ── Test 1: Held tensor + churn ───────────────────────────────────────
% After this, x must still be [1..100]. If the pool released x's
% buffer and a churn iteration reused it, x(50) would no longer be 50.
x = 1:100;
for k = 1:500
  y = (1:100) + k;  % same shape as x — best chance of pool reuse
end
assert(x(1) == 1, 'x(1) should be 1');
assert(x(50) == 50, 'x(50) should be 50');
assert(x(100) == 100, 'x(100) should be 100');

% ── Test 2: Held in cell + churn ──────────────────────────────────────
c = {1:100, ones(1, 100), zeros(1, 100)};
for k = 1:500
  z = (1:100) * 0.5;
  zz = ones(1, 100) - 1;
end
assert(c{1}(1) == 1 && c{1}(100) == 100, 'c{1} should be 1..100');
assert(c{2}(50) == 1, 'c{2} should be all ones');
assert(c{3}(50) == 0, 'c{3} should be all zeros');

% ── Test 3: Held in struct + churn ────────────────────────────────────
s.a = (1:100) * 2;
s.b = (1:100) + 1000;
for k = 1:500
  tmp = (1:100) - k;
end
assert(s.a(1) == 2 && s.a(100) == 200, 's.a unchanged');
assert(s.b(1) == 1001 && s.b(100) == 1100, 's.b unchanged');

% ── Test 4: Held by closure + churn ───────────────────────────────────
secret = (1:100) * 7;
get_secret = @() secret;
for k = 1:500
  q = (1:100) / k;
end
v = get_secret();
assert(v(1) == 7 && v(50) == 350 && v(100) == 700, 'closure sees intact secret');

% ── Test 5: Returned-from-function + caller churn ─────────────────────
function y = make_one_to_n(n)
  y = 1:n;
end
held = make_one_to_n(100);
for k = 1:500
  q = make_one_to_n(100);  % same size — high reuse pressure
end
assert(held(1) == 1 && held(50) == 50 && held(100) == 100, ...
  'held value from earlier call should be intact');

% ── Test 6: Multi-output function, both outputs held ─────────────────
function [a, b] = make_pair()
  a = ones(1, 100);
  b = ones(1, 100) * 2;
end
[hA, hB] = make_pair();
for k = 1:500
  [c, d] = make_pair();
  e = c + d;  % churn
end
assert(hA(1) == 1 && hA(100) == 1, 'hA still ones');
assert(hB(1) == 2 && hB(100) == 2, 'hB still twos');

% ── Test 7: Aliased read after another alias is destroyed ─────────────
% Bind two refs, drop one, churn, verify the other is intact.
a = (1:100) * 3;
b = a;     % alias
clear a;   % drop one ref
for k = 1:500
  q = (1:100) - 7;
end
assert(b(1) == 3 && b(50) == 150 && b(100) == 300, ...
  'survivor of an alias should be intact');

% ── Test 8: Held in a returned cell while caller churns ───────────────
function c = wrap_in_cell()
  c = { (1:100) * 11, (1:100) * 13 };
end
held_c = wrap_in_cell();
for k = 1:500
  tmp = wrap_in_cell();
end
assert(held_c{1}(1) == 11 && held_c{1}(100) == 1100, 'cell elem 1 intact');
assert(held_c{2}(1) == 13 && held_c{2}(100) == 1300, 'cell elem 2 intact');

% ── Test 9: Recursion with same-size local tensors ───────────────────
% The recursion stack creates many same-size locals; the unwind
% releases them in LIFO order. A buggy pool reuse during unwind
% would corrupt the lower stack frame's locals before they're read.
function r = sum_via_recursion(n)
  if n <= 0
    r = 0;
    return
  end
  buf = ones(1, 100) * n;  % same size every frame
  r = sum(buf) + sum_via_recursion(n - 1);
end
total = sum_via_recursion(50);
% total = sum_{k=1}^{50} 100*k = 100 * 50 * 51 / 2 = 127500
assert(total == 127500, sprintf('expected 127500, got %g', total));

% ── Test 10: Counter held across heavy alloc cycles ──────────────────
function h = make_counter()
  count = 0;
  function r = inc()
    count = count + 1;
    r = count;
  end
  h = @inc;
end
counter = make_counter();
for k = 1:300
  q = (1:100) + k;  % churn
  v = counter();
  assert(v == k, sprintf('counter reached %d but expected %d', v, k));
end

% ── Test 11: Comparison of fresh-vs-saved across reductions ──────────
% sum allocates a 1-element scratch internally (now released). Verify
% that releasing it doesn't break neighboring tensor data.
saved = (1:100) + 0.25;
for k = 1:300
  s_k = sum(1:100);
  assert(s_k == 5050, 'sum scalar wrong');
end
assert(abs(saved(50) - 50.25) < 1e-12, 'saved(50) corrupted by sum scratch');
assert(abs(saved(100) - 100.25) < 1e-12, 'saved(100) corrupted by sum scratch');

% ── Test 12: matmul churn while holding a previous matmul result ──────
A = ones(10) * 0.5;       % 10x10 all 0.5
held_M = A * A;            % each entry = 10 * 0.5 * 0.5 = 2.5
for k = 1:200
  M = A * A;               % churn — same shape
end
assert(held_M(1, 1) == 2.5 && held_M(10, 10) == 2.5, ...
  'held_M corrupted by matmul churn');

% ── Test 13: anon with heavy capture + churn ──────────────────────────
big = (1:100) * 19;
g = @(t) big(t);
for k = 1:300
  ww = (1:100) - k;
end
for i = 1:100
  v = g(i);
  assert(v == i * 19, sprintf('captured big(%d) wrong', i));
end

disp('SUCCESS')
