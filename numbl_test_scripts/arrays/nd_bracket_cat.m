% 3D tensor concatenation via bracket syntax [A B] and [A; B]

A = reshape(1:8, 2, 2, 2);
B = reshape(9:16, 2, 2, 2);

% ── Horizontal concatenation [A B] on 3D ─────────────────────────
C = [A B];
assert(size(C, 1) == 2)
assert(size(C, 2) == 4)
assert(size(C, 3) == 2)
% Page 1: A cols then B cols
assert(C(1, 1, 1) == A(1, 1, 1))
assert(C(1, 3, 1) == B(1, 1, 1))
assert(C(2, 4, 1) == B(2, 2, 1))
% Page 2
assert(C(1, 1, 2) == A(1, 1, 2))
assert(C(1, 3, 2) == B(1, 1, 2))

% ── Vertical concatenation [A; B] on 3D ──────────────────────────
D = [A; B];
assert(size(D, 1) == 4)
assert(size(D, 2) == 2)
assert(size(D, 3) == 2)
% Page 1: A rows then B rows
assert(D(1, 1, 1) == A(1, 1, 1))
assert(D(3, 1, 1) == B(1, 1, 1))
assert(D(4, 2, 1) == B(2, 2, 1))
% Page 2
assert(D(1, 1, 2) == A(1, 1, 2))
assert(D(3, 1, 2) == B(1, 1, 2))

disp('SUCCESS')
