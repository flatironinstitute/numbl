% Test: [C, H] = contour(X, Y, Z, levels, Name, Value).
% Verifies the contour matrix C, the contour handle H's properties, the
% parula(m) colormap matrix, and delete(H). Invariants hold in MATLAB too.

m = 41;
[xx, yy] = meshgrid(linspace(-1, 1, m));
zz = xx.^2 + yy.^2;             % level sets are circles of radius sqrt(level)
levels = [0.25 0.5 0.75];

[C, H] = contour(xx, yy, zz, levels, 'LineWidth', 2);

% --- Handle properties ---
assert(H.LineWidth == 2, 'LineWidth should be 2');
assert(strcmp(H.LineStyle, '-'), 'LineStyle should be -');
assert(isequal(H.LevelList, levels), 'LevelList should equal the levels');

% --- Contour matrix is well-formed: 2 rows, walkable headers ---
assert(size(C, 1) == 2, 'C must have 2 rows');
k = 1;
nlines = 0;
maxErr = 0;
while ( k < size(C, 2) )
    lvl = C(1, k);
    kl = C(2, k);
    assert(any(abs(lvl - levels) < 1e-9), 'header level must be in levels');
    v = k+1:k+kl;
    xv = C(1, v);
    yv = C(2, v);
    % Each vertex lies (approximately) on the circle x^2+y^2 = lvl.
    err = max(abs(xv.^2 + yv.^2 - lvl));
    maxErr = max(maxErr, err);
    nlines = nlines + 1;
    k = k + kl + 1;
end
assert(nlines > 0, 'expected at least one contour line');
assert(maxErr < 0.02, 'contour vertices should lie on the level set');

% --- parula(m) returns an m-by-3 colormap in [0,1] ---
cm = parula(8);
assert(isequal(size(cm), [8 3]), 'parula(8) should be 8x3');
assert(all(cm(:) >= 0 & cm(:) <= 1), 'colormap entries in [0,1]');

% --- delete(H) runs without error ---
delete(H);

disp('SUCCESS')
