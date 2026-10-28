% pcolor - Pseudocolor plot on a curvilinear grid
% A scalar field sampled on a polar grid, drawn as colored quadrilaterals.

[R, T] = meshgrid(linspace(0.5, 2, 20), linspace(0, pi, 30));
X = R .* cos(T);
Y = R .* sin(T);
C = sin(3 * T) .* R;

figure;
pcolor(X, Y, C);
shading flat;
colorbar;
title('Field on a Half Annulus');
axis equal;
