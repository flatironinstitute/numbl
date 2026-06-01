% Test: quiver3 input syntaxes and the returned Quiver handle.
% Values verified against MATLAB R2025b.

[X, Y] = meshgrid(-2:0.5:2, -2:0.5:2);
Z = X .* exp(-X.^2 - Y.^2);
U = ones(size(Z));
V = ones(size(Z));
W = 0.5 * ones(size(Z));

% --- Input syntaxes (must not error) ---
quiver3(X, Y, Z, U, V, W);          % full form
quiver3(Z, U, V, W);                 % Z-only form (grid from Z)
quiver3(X, Y, Z, U, V, W, 0);        % autoscale off
quiver3(X, Y, Z, U, V, W, 2);        % scale factor
quiver3(X, Y, Z, U, V, W, 'r');      % LineSpec color
quiver3(X, Y, Z, U, V, W, 'LineWidth', 2, 'ShowArrowHead', 'off');

% --- Handle from q = quiver3(...) ---
q = quiver3(X, Y, Z, U, V, W);
assert(strcmp(class(q), 'matlab.graphics.chart.primitive.Quiver'), ...
    'class(q) should be the Quiver class');
assert(q.LineWidth == 0.5, 'default LineWidth should be 0.5');
assert(q.AutoScaleFactor == 0.9, 'default AutoScaleFactor should be 0.9');

% --- Modify after creation (must not error) ---
q.ShowArrowHead = 'off';
q.Marker = '.';
q.LineWidth = 1.5;
assert(q.LineWidth == 1.5, 'LineWidth should update to 1.5');

% --- N-D arrays: one arrow per element (e.g. surfacefun stacks patches in
%     the 3rd dimension). Every element must become an arrow, not just the
%     first page. ---
X3 = zeros(2, 2, 3);
Y3 = zeros(2, 2, 3);
Z3 = zeros(2, 2, 3);
U3 = ones(2, 2, 3);
V3 = ones(2, 2, 3);
W3 = ones(2, 2, 3);
q3 = quiver3(X3, Y3, Z3, U3, V3, W3, 0);
assert(numel(q3.UData) == 12, 'all 12 elements should be arrows');
assert(numel(q3.XData) == 12, 'base x-coordinates cover all elements');

disp('SUCCESS')
