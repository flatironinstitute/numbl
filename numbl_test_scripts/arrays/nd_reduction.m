% 3D tensor reductions: sum, prod, mean along various dimensions

a = reshape(1:24, 2, 3, 4);

% ── sum with no dim arg on 3D tensor ────────────────────────────────
% MATLAB default: reduces along first non-singleton dimension (dim 1)
b = sum(a);
assert(size(b, 1) == 1)
assert(size(b, 2) == 3)
assert(size(b, 3) == 4)
assert(b(1, 1, 1) == 3)   % 1+2
assert(b(1, 2, 1) == 7)   % 3+4
assert(b(1, 3, 1) == 11)  % 5+6

% ── sum along dim 1 ─────────────────────────────────────────────────
c = sum(a, 1);
assert(size(c, 1) == 1)
assert(size(c, 2) == 3)
assert(c(1, 1, 1) == 3)

% ── sum along dim 2 ─────────────────────────────────────────────────
d = sum(a, 2);
assert(size(d, 1) == 2)
assert(size(d, 2) == 1)
assert(size(d, 3) == 4)
assert(d(1, 1, 1) == 9)    % 1+3+5
assert(d(2, 1, 1) == 12)   % 2+4+6

% ── sum along dim 3 ─────────────────────────────────────────────────
e = sum(a, 3);
assert(size(e, 1) == 2)
assert(size(e, 2) == 3)
% e(1,1) = a(1,1,1)+a(1,1,2)+a(1,1,3)+a(1,1,4) = 1+7+13+19 = 40
assert(e(1, 1) == 40)
% e(2,1) = a(2,1,1)+a(2,1,2)+a(2,1,3)+a(2,1,4) = 2+8+14+20 = 44
assert(e(2, 1) == 44)

disp('SUCCESS')
