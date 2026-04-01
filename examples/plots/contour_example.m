% contour - 2D contour plot
% Demonstrates contour lines of a two-peaked surface

x = linspace(-3, 3, 80);
y = linspace(-3, 3, 80);
[X, Y] = meshgrid(x, y);
Z = exp(-((X - 1).^2 + Y.^2)) + 0.8 * exp(-((X + 1).^2 + Y.^2));

figure;
contour(X, Y, Z);
title('Contour Lines of Two Peaks');
xlabel('X');
ylabel('Y');
colorbar;
