% Test gradient function on 2D matrix
% In MATLAB, [fx, fy] = gradient(F) returns two matrices:
% fx = gradient in x-direction (across columns)
% fy = gradient in y-direction (across rows)

% Basic 2x3 matrix
F = [1 2 4; 3 5 9];
[fx, fy] = gradient(F);

% Both outputs should be same size as input
assert(size(fx, 1) == 2);
assert(size(fx, 2) == 3);
assert(size(fy, 1) == 2);
assert(size(fy, 2) == 3);

% fx: gradient along columns (x-direction)
% Row 1: [2-1, (4-1)/2, 4-2] = [1, 1.5, 2]
% Row 2: [5-3, (9-3)/2, 9-5] = [2, 3, 4]
assert(abs(fx(1,1) - 1) < 1e-10);
assert(abs(fx(1,2) - 1.5) < 1e-10);
assert(abs(fx(1,3) - 2) < 1e-10);
assert(abs(fx(2,1) - 2) < 1e-10);
assert(abs(fx(2,2) - 3) < 1e-10);
assert(abs(fx(2,3) - 4) < 1e-10);

% fy: gradient along rows (y-direction)
% Col 1: [3-1] = [2; 2]
% Col 2: [5-2] = [3; 3]
% Col 3: [9-4] = [5; 5]
assert(abs(fy(1,1) - 2) < 1e-10);
assert(abs(fy(2,1) - 2) < 1e-10);
assert(abs(fy(1,2) - 3) < 1e-10);
assert(abs(fy(2,2) - 3) < 1e-10);
assert(abs(fy(1,3) - 5) < 1e-10);
assert(abs(fy(2,3) - 5) < 1e-10);

% Single output: should return fx only
g = gradient(F);
assert(isequal(g, fx));

% 1D vector gradient
v = [1 3 6 10];
g2 = gradient(v);
% [3-1, (6-1)/2, (10-3)/2, 10-6] = [2, 2.5, 3.5, 4]
assert(abs(g2(1) - 2) < 1e-10);
assert(abs(g2(2) - 2.5) < 1e-10);
assert(abs(g2(3) - 3.5) < 1e-10);
assert(abs(g2(4) - 4) < 1e-10);

% 3x3 matrix
F2 = [1 2 3; 4 5 6; 7 8 9];
[gx, gy] = gradient(F2);
assert(size(gx, 1) == 3);
assert(size(gx, 2) == 3);
assert(size(gy, 1) == 3);
assert(size(gy, 2) == 3);

disp('SUCCESS');
