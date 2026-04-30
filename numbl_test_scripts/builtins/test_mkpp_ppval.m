% Test mkpp and ppval builtins

% --- Scalar pp with three pieces of mixed orders ---
breaks = [0 4 10 15];
coefs = [0 1 -1 1 1; 0 0 1 -2 53; -1 6 1 4 77];
pp = mkpp(breaks, coefs);

assert(strcmp(pp.form, 'pp'));
assert(isequal(pp.breaks, [0 4 10 15]));
assert(isequal(pp.coefs, coefs));
assert(pp.pieces == 3);
assert(pp.order == 5);
assert(pp.dim == 1);

% Piece 1 on [0,4]: f(x) = 0*x^4 + 1*x^3 - x^2 + x + 1
assert(abs(ppval(pp, 0) - 1) < 1e-12);
assert(abs(ppval(pp, 2) - (8 - 4 + 2 + 1)) < 1e-12);

% Piece 2 on [4,10]: f(x) = (x-4)^2 - 2*(x-4) + 53
% f(4) = 53, f(10) = 36 - 12 + 53 = 77
assert(abs(ppval(pp, 4) - 53) < 1e-12);
assert(abs(ppval(pp, 10) - 77) < 1e-12);

% Piece 3 on [10,15]: f(x) = -1*(x-10)^4 + 6*(x-10)^3 + (x-10)^2 + 4*(x-10) + 77
% f(10) = 77, f(15) = -625 + 750 + 25 + 20 + 77 = 247
assert(abs(ppval(pp, 15) - 247) < 1e-12);

% Vector query
xq = [0 2 4 10 15];
yq = ppval(pp, xq);
expected = [1, 7, 53, 77, 247];
assert(norm(yq - expected) < 1e-10);
assert(isequal(size(yq), size(xq)));

% --- Single-piece pp (mkpp on [-8, -4] with cc = [-1/4 1 0]) ---
cc = [-1/4 1 0];
pp1 = mkpp([-8 -4], cc);
assert(pp1.pieces == 1);
assert(pp1.order == 3);

% f(x) = -1/4 * (x - (-8))^2 + (x - (-8)) at -8 = 0
assert(abs(ppval(pp1, -8) - 0) < 1e-12);
% at -4: -1/4 * 16 + 4 = -4 + 4 = 0
assert(abs(ppval(pp1, -4) - 0) < 1e-12);
% at -6: -1/4 * 4 + 2 = -1 + 2 = 1
assert(abs(ppval(pp1, -6) - 1) < 1e-12);

% --- Multi-piece alternating ---
pp2 = mkpp([-8 -4 0 4 8], [cc; -cc; cc; -cc]);
assert(pp2.pieces == 4);
assert(pp2.order == 3);

% Sample some interior points and verify continuity at break points
assert(abs(ppval(pp2, -4) - 0) < 1e-12);
assert(abs(ppval(pp2, 0) - 0) < 1e-12);
assert(abs(ppval(pp2, 4) - 0) < 1e-12);

% --- Column vector query preserves orientation ---
xqc = [0; 2; 4];
yqc = ppval(pp, xqc);
assert(isequal(size(yqc), [3 1]));
assert(abs(yqc(1) - 1) < 1e-12);
assert(abs(yqc(3) - 53) < 1e-12);

% --- Vector-valued (dim > 1) ---
% Two pieces on [0,1,2], dim = 2.
% Piece 1: f1(x) = x + 1 (rows 1: [1 1]), f2(x) = 2*x (rows 2: [2 0])
% Piece 2: f1(x) = 1 - (x-1) (rows 3: [-1 1]), f2(x) = 2 (rows 4: [0 2])
coefV = [1 1; 2 0; -1 1; 0 2];
ppv = mkpp([0 1 2], coefV, 2);
assert(ppv.dim == 2);
assert(ppv.pieces == 2);
assert(ppv.order == 2);

vv = ppval(ppv, 0);
assert(isequal(size(vv), [2 1]));
assert(abs(vv(1) - 1) < 1e-12);
assert(abs(vv(2) - 0) < 1e-12);

vv = ppval(ppv, 1);
% piece 2 chosen at break: f1(1) = 1, f2(1) = 2
assert(abs(vv(1) - 1) < 1e-12);
assert(abs(vv(2) - 2) < 1e-12);

% Vector query for vector-valued pp produces [d, N]
vv2 = ppval(ppv, [0 0.5 1 1.5 2]);
assert(isequal(size(vv2), [2 5]));
assert(abs(vv2(1, 1) - 1) < 1e-12);
assert(abs(vv2(2, 1) - 0) < 1e-12);
assert(abs(vv2(2, 5) - 2) < 1e-12);

disp('SUCCESS');
