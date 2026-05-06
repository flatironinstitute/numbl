% Stress tests targeting buffer-aliasing scenarios where the memory
% pool could cause inaccurate results. Each test sets up a held value
% that should NOT be touched by surrounding operations, then verifies
% the held value is unchanged after heavy pool activity.
%
% If the pool releases a held buffer prematurely, subsequent same-size
% allocations would reuse it. Pool re-use zero-fills on acquire, so
% the held tensor would suddenly read as zeros — the asserts catch it.

% ── Test 1: Indexed read after copy (COW aliasing) ────────────────────
% `b = a` aliases. `a(2) = 99` should COW so b is untouched.
% After many allocations of the same size, b should still hold the
% original values.
a = (1:50) + 0.5;
b = a;            % alias
a(2) = 999;       % mutates a but not b (COW)
for k = 1:500
  q = (1:50) + k;
end
assert(b(1) == 1.5 && b(2) == 2.5 && b(50) == 50.5, ...
  'b should be unchanged by a''s mutation and by churn');
assert(a(2) == 999 && a(1) == 1.5 && a(50) == 50.5, ...
  'a''s mutation should be intact');

% ── Test 2: Cell aliasing ─────────────────────────────────────────────
% Cell stores a tensor by reference. Modifying via cell index COWs.
v = (1:50) * 11;
c = {v};
c{1}(2) = -1;    % mutate c{1}, not v
for k = 1:500
  q = (1:50) * k;
end
assert(v(2) == 22, 'v unchanged by c{1} mutation');
assert(c{1}(2) == -1, 'c{1}(2) should be -1');
assert(v(50) == 550, 'v(50) unchanged');
assert(c{1}(50) == 550, 'c{1}(50) unchanged');

% ── Test 3: Struct field aliasing ─────────────────────────────────────
v = (1:50) * 7;
s.x = v;
s.x(3) = 9999;
for k = 1:500
  q = (1:50) - k;
end
assert(v(3) == 21, 'v(3) unchanged');
assert(s.x(3) == 9999, 's.x(3) mutated');
assert(v(50) == 350 && s.x(50) == 350, 'unmutated indices match');

% ── Test 4: Nested cell aliasing ──────────────────────────────────────
v = (1:50) + 0.1;
c = {{v}};
c{1}{1}(5) = 88;
for k = 1:500
  q = (1:50);
end
assert(v(5) == 5.1, 'v(5) unchanged after nested cell mutation');
assert(c{1}{1}(5) == 88, 'c{1}{1}(5) should be 88');

% ── Test 5: Function returns alias of param + caller mutation ─────────
function y = pass_through(x)
  y = x;          % returns alias
end
p = (1:50) * 2;
q = pass_through(p);  % q aliases p (no copy)
p(1) = -1;            % mutate p, q should be unchanged
for k = 1:500
  tmp = (1:50) - k;
end
assert(q(1) == 2, 'q(1) should be 2 (pre-mutation alias of p)');
assert(p(1) == -1, 'p(1) should be -1');

% ── Test 6: Anonymous fn snapshot vs caller mutation ──────────────────
% MATLAB's anonymous functions snapshot their workspace by value at
% definition time. After defining f, mutating `captured` must NOT
% affect what f sees. (Implementation: the snapshot env is exposed to
% the alias sweep so parent-side index stores trigger COW.)
captured = (1:50) * 3;
f = @(t) captured(t) * 2;
captured(1) = -50;   % must NOT change what f sees
for k = 1:500
  q = (1:50) + k;
end
v = f(1);
assert(v == 6, sprintf('anon snapshot: expected 6, got %g', v));
v50 = f(50);
assert(v50 == 50 * 3 * 2, 'snapshot at index 50 should be unchanged');
% And the original `captured` should still reflect the post-snapshot mutation.
assert(captured(1) == -50, 'captured(1) should remain -50 after snapshot');

% ── Test 7: Re-use across function boundaries ─────────────────────────
% Function returns a buffer; caller binds to slot. If caller's churn
% reuses the buffer, the slot's data corrupts.
function y = make_pattern(n)
  y = zeros(1, n);
  for i = 1:n; y(i) = i * 17; end;
end
held1 = make_pattern(50);
held2 = make_pattern(50);
held3 = make_pattern(50);
% Now churn — each iteration creates and discards a same-size buffer.
for k = 1:500
  q = make_pattern(50);
end
assert(held1(25) == 25*17 && held1(50) == 50*17, 'held1 corrupted');
assert(held2(25) == 25*17 && held2(50) == 50*17, 'held2 corrupted');
assert(held3(25) == 25*17 && held3(50) == 50*17, 'held3 corrupted');

% ── Test 8: COW on growing assignment ────────────────────────────────
% a(end+1) = x should COW; b shouldn't see the new element.
a = [10 20 30];
b = a;
a(4) = 40;
for k = 1:500
  q = [k k+1 k+2];
end
assert(length(b) == 3, 'b should still be length 3');
assert(b(1) == 10 && b(2) == 20 && b(3) == 30, 'b values unchanged');
assert(length(a) == 4 && a(4) == 40, 'a grew correctly');

% ── Test 9: Tight COW + indexed mutation in loop ──────────────────────
% Each iteration: alias, mutate, drop. Many same-size allocs.
final_b = zeros(1, 50);
for k = 1:200
  a = (1:50) + k;
  b = a;
  a(1) = -k;
  if k == 200
    final_b = b;
  end
end
% At k=200, b was a copy/alias of (1:50) + 200, then a(1)=-200 should
% have COW'd a, leaving b intact.
assert(final_b(1) == 201, sprintf('expected 201, got %g', final_b(1)));
assert(final_b(50) == 250, 'final_b(50) should be 250');

% ── Test 10: Logical-mask read with churn ────────────────────────────
data = (1:100) * 1.5;
mask = data > 75;
filtered = data(mask);
held = filtered;
for k = 1:500
  q = data > k;       % new logical tensor each iter
  r = data(q);        % new filtered tensor each iter
end
assert(length(held) == 50, sprintf('expected 50, got %d', length(held)));
assert(held(1) == 76.5, sprintf('held(1) expected 76.5, got %g', held(1)));
assert(held(50) == 150, 'held(50) should be 150');

% ── Test 11: Complex tensor + churn ──────────────────────────────────
z = (1:100) + (1:100) * 1i;
held = z * 2;
for k = 1:500
  q = (1:100) + k * 1i;
end
assert(real(held(1)) == 2 && imag(held(1)) == 2, 'held(1) corrupted');
assert(real(held(50)) == 100 && imag(held(50)) == 100, 'held(50) corrupted');

% ── Test 12: Sparse + churn ──────────────────────────────────────────
S = sparse([1 2 3 4 5], [1 2 3 4 5], (1:5) * 100);
held_S = S;
for k = 1:200
  T = sparse([1 2 3], [1 2 3], (1:3) + k);
end
% held_S should still represent diag([100 200 300 400 500])
assert(held_S(1, 1) == 100, 'held_S(1,1) corrupted');
assert(held_S(5, 5) == 500, 'held_S(5,5) corrupted');

% ── Test 13: Reduction scratch + held vector ─────────────────────────
held = (1:1000) * 0.001;
for k = 1:500
  s_k = sum(held);    % held used as input; reduction allocs scratch
  m_k = mean(held);
  mn_k = min(held);
  mx_k = max(held);
end
assert(abs(held(1) - 0.001) < 1e-12, 'held(1) corrupted by reduction scratch');
assert(abs(held(500) - 0.5) < 1e-12, 'held(500) corrupted');
assert(abs(held(1000) - 1.0) < 1e-12, 'held(1000) corrupted');

% ── Test 14: linalg result held + linalg churn ────────────────────────
A = [1 2; 3 4];
held_inv = inv(A);
for k = 1:100
  Q = inv(A + k * eye(2));
  R = inv(Q);
end
% held_inv should be inv([1 2; 3 4]) = [-2 1; 1.5 -0.5]
assert(abs(held_inv(1, 1) - (-2)) < 1e-10, 'held_inv(1,1) corrupted');
assert(abs(held_inv(2, 2) - (-0.5)) < 1e-10, 'held_inv(2,2) corrupted');

disp('SUCCESS')
