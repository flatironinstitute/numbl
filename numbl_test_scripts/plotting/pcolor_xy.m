% pcolor with explicit X/Y vertex coordinates from meshgrid.

x = linspace(-2, 2, 30);
y = linspace(-1, 1, 20);
[X, Y] = meshgrid(x, y);
Z = sin(X) .* cos(Y);

pcolor(X, Y, Z);
colorbar;
shading flat;
title('pcolor with meshgrid');

disp('SUCCESS');
