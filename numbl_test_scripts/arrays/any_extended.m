% Tests for extended any() syntax: any(A,'all'), any(A,dim), any(A,vecdim)

% ── any(A,'all') ─────────────────────────────────────────────────────────

% all-zero matrix -> false
A = [0 0; 0 0];
assert(any(A,'all') == 0)

% matrix with one nonzero -> true
B = [0 0; 0 1];
assert(any(B,'all') == 1)

% row vector
assert(any([0 0 0],'all') == 0)
assert(any([0 1 0],'all') == 1)

% column vector
assert(any([0;0;0],'all') == 0)
assert(any([1;0;0],'all') == 1)

% scalar
assert(any(0,'all') == 0)
assert(any(5,'all') == 1)

% ── any(A,dim) ───────────────────────────────────────────────────────────

% any(A,1): reduce along rows -> row vector result
M = [0 1; 0 0; 1 0];
R1 = any(M, 1);
% col1: any([0;0;1])=1, col2: any([1;0;0])=1
assert(size(R1,1) == 1)
assert(size(R1,2) == 2)
assert(R1(1) == 1)
assert(R1(2) == 1)

% any(A,2): reduce along columns -> column vector result
R2 = any(M, 2);
% row1: any([0 1])=1, row2: any([0 0])=0, row3: any([1 0])=1
assert(size(R2,1) == 3)
assert(size(R2,2) == 1)
assert(R2(1) == 1)
assert(R2(2) == 0)
assert(R2(3) == 1)

% any on row vector with dim=2 -> scalar
assert(any([0 1 0], 2) == 1)
assert(any([0 0 0], 2) == 0)

% any on row vector with dim=1 -> unchanged (dim 1 has size 1)
rv = any([0 1 0], 1);
assert(size(rv,1) == 1)
assert(size(rv,2) == 3)
assert(rv(1) == 0)
assert(rv(2) == 1)
assert(rv(3) == 0)

% any on column vector with dim=1 -> scalar
assert(any([1;0;1], 1) == 1)
assert(any([0;0;0], 1) == 0)

% any on column vector with dim=2 -> unchanged (dim 2 has size 1)
cv = any([1;0;1], 2);
assert(size(cv,1) == 3)
assert(size(cv,2) == 1)
assert(cv(1) == 1)
assert(cv(2) == 0)
assert(cv(3) == 1)

% ── any(A,vecdim) ────────────────────────────────────────────────────────

% any(A,[1 2]) on matrix: equivalent to any(A,'all')
N = [0 0; 1 0];
assert(any(N,[1 2]) == 1)

N2 = [0 0; 0 0];
assert(any(N2,[1 2]) == 0)

disp('SUCCESS')
