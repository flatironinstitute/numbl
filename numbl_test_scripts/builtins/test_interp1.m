% Test interp1 - 1D interpolation

% Linear interpolation (default)
x = [1 2 3 4 5];
y = [2 4 6 8 10];
assert(abs(interp1(x, y, 2.5) - 5) < 1e-10);
assert(abs(interp1(x, y, 1) - 2) < 1e-10);
assert(abs(interp1(x, y, 5) - 10) < 1e-10);

% Query at multiple points
xq = [1.5 2.5 3.5];
yq = interp1(x, y, xq);
assert(norm(yq - [3 5 7]) < 1e-10);

% Non-uniform spacing
x2 = [0 1 3 6];
y2 = [0 1 3 6];
assert(abs(interp1(x2, y2, 2) - 2) < 1e-10);
assert(abs(interp1(x2, y2, 4.5) - 4.5) < 1e-10);

% Quadratic data with linear interp
x3 = [0 1 2 3 4];
y3 = [0 1 4 9 16];
v = interp1(x3, y3, 1.5);
assert(abs(v - 2.5) < 1e-10);

% Nearest interpolation
x4 = [1 2 3 4];
y4 = [10 20 30 40];
assert(abs(interp1(x4, y4, 1.3, 'nearest') - 10) < 1e-10);
assert(abs(interp1(x4, y4, 1.6, 'nearest') - 20) < 1e-10);
assert(abs(interp1(x4, y4, 2.5, 'nearest') - 30) < 1e-10);

% Extrapolation returns NaN by default
v_extrap = interp1(x, y, 6);
assert(isnan(v_extrap));

% Extrapolation with 'extrap'
v_extrap2 = interp1(x, y, 6, 'linear', 'extrap');
assert(abs(v_extrap2 - 12) < 1e-10);

% Column vector output preserved
xq_col = [1.5; 2.5; 3.5];
yq_col = interp1(x, y, xq_col);
assert(size(yq_col, 1) == 3);
assert(size(yq_col, 2) == 1);

% Matrix y: each column interpolated independently against x.
% Result is length(xq)-by-size(y,2).
xm = (1:4)';
ym = [xm, 2*xm, 3*xm];        % 4x3, columns are 1x, 2x, 3x
ynew = interp1(1:4, ym, linspace(1, 4, 7));
assert(isequal(size(ynew), [7 3]), 'matrix-y interp1 result is nq x ncols');
assert(norm(ynew(:, 1) - linspace(1, 4, 7)') < 1e-10, 'col 1');
assert(norm(ynew(:, 2) - 2 * linspace(1, 4, 7)') < 1e-10, 'col 2');
assert(norm(ynew(:, 3) - 3 * linspace(1, 4, 7)') < 1e-10, 'col 3');

% Downsampling a colormap-like matrix (the cmocean use case).
cmap = rand(256, 3);
small = interp1(1:256, cmap, linspace(1, 256, 64));
assert(isequal(size(small), [64 3]), 'downsampled colormap size');

disp('SUCCESS');
