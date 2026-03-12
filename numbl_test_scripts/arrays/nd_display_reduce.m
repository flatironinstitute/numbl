% 3D tensor display and additional reductions (prod, mean along dim 3)

a = reshape(1:12, 2, 3, 2);

% ── disp on 3D tensor should not crash ────────────────────────────
disp(a)

% ── prod along dim 3 ──────────────────────────────────────────────
b = prod(a, 3);
assert(size(b, 1) == 2)
assert(size(b, 2) == 3)
% b(1,1) = a(1,1,1) * a(1,1,2) = 1 * 7 = 7
assert(b(1, 1) == 7)
% b(2,1) = a(2,1,1) * a(2,1,2) = 2 * 8 = 16
assert(b(2, 1) == 16)
% b(1,3) = a(1,3,1) * a(1,3,2) = 5 * 11 = 55
assert(b(1, 3) == 55)

% ── mean along dim 1 on 3D ────────────────────────────────────────
c = mean(a, 1);
assert(size(c, 1) == 1)
assert(size(c, 2) == 3)
assert(size(c, 3) == 2)
% c(1,1,1) = mean([1, 2]) = 1.5
assert(c(1, 1, 1) == 1.5)

% ── mean along dim 3 on 3D ────────────────────────────────────────
d = mean(a, 3);
assert(size(d, 1) == 2)
assert(size(d, 2) == 3)
% d(1,1) = mean([1, 7]) = 4
assert(d(1, 1) == 4)
% d(2,3) = mean([6, 12]) = 9
assert(d(2, 3) == 9)

disp('SUCCESS')
