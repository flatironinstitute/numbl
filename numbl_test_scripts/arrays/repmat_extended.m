% Extended repmat tests: all calling conventions

% ── repmat(A, n) — scalar n means repeat n times in rows and columns ──

% Scalar input with scalar n
B = repmat(10, 3, 2);
assert(all(size(B) == [3 2]));
assert(all(all(B == 10)));

% Matrix input with scalar n (square block)
A = [1 2; 3 4];
B = repmat(A, 2);
assert(all(size(B) == [4 4]));
assert(B(1,1) == 1); assert(B(1,2) == 2); assert(B(1,3) == 1); assert(B(1,4) == 2);
assert(B(2,1) == 3); assert(B(2,2) == 4); assert(B(2,3) == 3); assert(B(2,4) == 4);
assert(B(3,1) == 1); assert(B(3,2) == 2); assert(B(3,3) == 1); assert(B(3,4) == 2);
assert(B(4,1) == 3); assert(B(4,2) == 4); assert(B(4,3) == 3); assert(B(4,4) == 4);

% ── repmat(A, r1, ..., rN) — already supported, but verify ───────────

% 2D repetition
A = [1 2; 3 4];
B = repmat(A, 2, 3);
assert(all(size(B) == [4 6]));
assert(B(1,1) == 1); assert(B(1,3) == 1); assert(B(1,5) == 1);
assert(B(3,1) == 1); assert(B(3,3) == 1); assert(B(3,5) == 1);

% 3D repetition via separate args
A = [1 2; 3 4];
B = repmat(A, 2, 3, 2);
assert(size(B, 1) == 4);
assert(size(B, 2) == 6);
assert(size(B, 3) == 2);

% ── repmat(A, r) — vector r specifies repetition scheme ──────────────

% Row vector r
A = [1 2; 3 4];
B = repmat(A, [2 3]);
assert(all(size(B) == [4 6]));
assert(B(1,1) == 1); assert(B(1,3) == 1); assert(B(1,5) == 1);
assert(B(3,1) == 1); assert(B(3,3) == 1);

% 3D via vector
A = [1 2; 3 4];
B = repmat(A, [2 3 2]);
assert(size(B, 1) == 4);
assert(size(B, 2) == 6);
assert(size(B, 3) == 2);

% ── repmat with scalar input ─────────────────────────────────────────

% Scalar replicated with vector r
B = repmat(5, [2 3]);
assert(all(size(B) == [2 3]));
assert(all(all(B == 5)));

% Scalar replicated with scalar n
B = repmat(5, 3);
assert(all(size(B) == [3 3]));
assert(all(all(B == 5)));

% ── Vertical/horizontal stacking ─────────────────────────────────────

% Vertical stack of row vector
A = 1:4;
B = repmat(A, 4, 1);
assert(all(size(B) == [4 4]));
assert(B(1,1) == 1); assert(B(4,4) == 4);
assert(B(2,3) == 3); assert(B(3,2) == 2);

% Horizontal stack of column vector
A = (1:3)';
B = repmat(A, 1, 4);
assert(all(size(B) == [3 4]));
assert(B(1,1) == 1); assert(B(1,4) == 1);
assert(B(3,1) == 3); assert(B(3,4) == 3);

% ── repmat(A, 1) — identity operation ────────────────────────────────
A = [1 2; 3 4];
B = repmat(A, 1);
assert(all(size(B) == [2 2]));
assert(B(1,1) == 1); assert(B(1,2) == 2);
assert(B(2,1) == 3); assert(B(2,2) == 4);

disp('SUCCESS');
