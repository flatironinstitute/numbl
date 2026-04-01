% mesh - Mesh surface plot
% Demonstrates a wireframe mesh of a saddle surface

x = linspace(-2, 2, 40);
y = linspace(-2, 2, 40);
[X, Y] = meshgrid(x, y);
Z = X.^2 - Y.^2;

figure;
mesh(X, Y, Z);
title('Saddle Surface (Hyperbolic Paraboloid)');
xlabel('X');
ylabel('Y');
zlabel('Z');
