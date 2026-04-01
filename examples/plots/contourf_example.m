% contourf - Filled contour plot
% Demonstrates filled contours of a ring-shaped function

x = linspace(-3, 3, 80);
y = linspace(-3, 3, 80);
[X, Y] = meshgrid(x, y);
R = sqrt(X.^2 + Y.^2);
Z = sin(R) ./ (R + 0.1);

figure;
contourf(X, Y, Z);
title('Filled Contours of a Ripple Function');
xlabel('X');
ylabel('Y');
colorbar;
