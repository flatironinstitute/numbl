% Indexed assignment with slices on 3D tensors

A = reshape(1:12, 2, 3, 2);

% ── Assign scalar to single element ──────────────────────────────
A(1,1,1) = 99;
assert(A(1,1,1) == 99)
assert(A(2,1,1) == 2)  % others unchanged

% ── Assign to entire column of a page ────────────────────────────
A(1,1,1) = 1;  % restore
B = reshape(1:8, 2, 2, 2);
B(:, 1, 2) = [50; 60];
assert(B(1,1,2) == 50)
assert(B(2,1,2) == 60)
assert(B(1,2,2) == 7)  % others unchanged

% ── Assign to a full page (slice along dim 3) ────────────────────
C = reshape(1:12, 2, 3, 2);
C(:,:,2) = [10 30 50; 20 40 60];
assert(C(1,1,2) == 10)
assert(C(2,1,2) == 20)
assert(C(1,2,2) == 30)
assert(C(2,3,2) == 60)
% Page 1 unchanged
assert(C(1,1,1) == 1)

% ── Assign with colon to all elements ────────────────────────────
D = zeros(2, 2, 2);
D(:) = 1:8;
assert(D(1,1,1) == 1)
assert(D(2,2,2) == 8)

disp('SUCCESS')
