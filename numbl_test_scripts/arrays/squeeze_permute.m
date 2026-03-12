% squeeze and permute on 3D tensors

% ── squeeze removes singleton dimensions ─────────────────────────
A = reshape(1:6, 1, 3, 2);
B = squeeze(A);
% [1,3,2] → [3,2]
assert(size(B, 1) == 3)
assert(size(B, 2) == 2)
assert(B(1,1) == 1)
assert(B(3,2) == 6)

% ── squeeze on [2,1,3] → [2,3] ──────────────────────────────────
C = reshape(1:6, 2, 1, 3);
D = squeeze(C);
assert(size(D, 1) == 2)
assert(size(D, 2) == 3)
assert(D(1,1) == 1)
assert(D(2,3) == 6)

% ── permute: swap dimensions ─────────────────────────────────────
E = reshape(1:12, 2, 3, 2);
F = permute(E, [2 1 3]);
% [2,3,2] → [3,2,2]
assert(size(F, 1) == 3)
assert(size(F, 2) == 2)
assert(size(F, 3) == 2)
% E(1,1,1)=1 → F(1,1,1)=1
assert(F(1,1,1) == 1)
% E(2,1,1)=2 → F(1,2,1)=2
assert(F(1,2,1) == 2)
% E(1,2,1)=3 → F(2,1,1)=3
assert(F(2,1,1) == 3)

disp('SUCCESS')
