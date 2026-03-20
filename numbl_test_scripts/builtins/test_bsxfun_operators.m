% Test bsxfun fast-path for known operator handles.
% Ensures results match direct element-wise operations.

%% @times: complex matrix * complex column
A = [1+2i 3+4i; 5+6i 7+8i];
v = [10+1i; 20+2i];
r1 = bsxfun(@times, A, v);
r2 = A .* v;
assert(max(abs(r1(:) - r2(:))) < 1e-12, 'bsxfun @times complex col should match .*');

%% @times: complex matrix * complex row
w = [10+1i 20+2i];
r1 = bsxfun(@times, A, w);
r2 = A .* w;
assert(max(abs(r1(:) - r2(:))) < 1e-12, 'bsxfun @times complex row should match .*');

%% @times: real matrix * real column
Ar = [1 2; 3 4; 5 6];
vr = [10; 20; 30];
r1 = bsxfun(@times, Ar, vr);
r2 = Ar .* vr;
assert(all(r1(:) == r2(:)), 'bsxfun @times real col');

%% @times: real matrix * real row
wr = [10 20];
r1 = bsxfun(@times, Ar, wr);
r2 = Ar .* wr;
assert(all(r1(:) == r2(:)), 'bsxfun @times real row');

%% @plus: complex broadcast
r1 = bsxfun(@plus, A, v);
r2 = A + v;
assert(max(abs(r1(:) - r2(:))) < 1e-12, 'bsxfun @plus complex');

%% @minus: complex broadcast
r1 = bsxfun(@minus, A, v);
r2 = A - v;
assert(max(abs(r1(:) - r2(:))) < 1e-12, 'bsxfun @minus complex');

%% @rdivide: complex broadcast
r1 = bsxfun(@rdivide, A, v);
r2 = A ./ v;
assert(max(abs(r1(:) - r2(:))) < 1e-12, 'bsxfun @rdivide complex');

%% @plus: real broadcast
r1 = bsxfun(@plus, Ar, vr);
r2 = Ar + vr;
assert(all(r1(:) == r2(:)), 'bsxfun @plus real');

%% @minus: real broadcast
r1 = bsxfun(@minus, Ar, vr);
r2 = Ar - vr;
assert(all(r1(:) == r2(:)), 'bsxfun @minus real');

%% @rdivide: real broadcast
r1 = bsxfun(@rdivide, Ar, vr);
r2 = Ar ./ vr;
assert(max(abs(r1(:) - r2(:))) < 1e-12, 'bsxfun @rdivide real');

%% Scalar broadcast
r1 = bsxfun(@times, Ar, 5);
r2 = Ar .* 5;
assert(all(r1(:) == r2(:)), 'bsxfun @times scalar');

%% 3D broadcast
B = randn(3,4,5);
u = randn(3,1,5);
r1 = bsxfun(@times, B, u);
r2 = B .* u;
assert(max(abs(r1(:) - r2(:))) < 1e-12, 'bsxfun @times 3D broadcast');

r1 = bsxfun(@plus, B, u);
r2 = B + u;
assert(max(abs(r1(:) - r2(:))) < 1e-12, 'bsxfun @plus 3D broadcast');

%% Same-size (no broadcast needed)
C = randn(4,4);
D = randn(4,4);
r1 = bsxfun(@times, C, D);
r2 = C .* D;
assert(max(abs(r1(:) - r2(:))) < 1e-12, 'bsxfun @times same size');

%% Mixed real/complex
vc = [1+1i; 2+2i; 3+3i];
r1 = bsxfun(@times, Ar, vc);
r2 = Ar .* vc;
assert(max(abs(r1(:) - r2(:))) < 1e-12, 'bsxfun @times mixed real*complex');

%% Non-operator handle should still work (falls through to generic path)
r1 = bsxfun(@(a,b) a.*b + 1, Ar, vr);
expected = Ar .* vr + 1;
assert(max(abs(r1(:) - expected(:))) < 1e-12, 'bsxfun with anonymous handle');

%% Larger sizes (closer to chebfun usage)
M = complex(randn(133,133), randn(133,133));
col = complex(randn(133,1), randn(133,1));
r1 = bsxfun(@times, M, col);
r2 = M .* col;
assert(max(abs(r1(:) - r2(:))) < 1e-10, 'bsxfun @times large complex');

disp('SUCCESS');
