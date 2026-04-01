% surf - 3D surface plot
% Demonstrates a surface plot of a 2D Gaussian

x = linspace(-3, 3, 50);
y = linspace(-3, 3, 50);
[X, Y] = meshgrid(x, y);
Z = exp(-(X.^2 + Y.^2) / 2);

figure;
surf(X, Y, Z);
title('2D Gaussian Surface');
xlabel('X');
ylabel('Y');
zlabel('Z');
colorbar;
