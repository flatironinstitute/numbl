% quiver - 2D vector field plot
% A point vortex superimposed on a uniform horizontal stream.

[X, Y] = meshgrid(linspace(-2, 2, 16));
r2 = X.^2 + Y.^2 + 0.1;
U = 1 - Y ./ r2;
V = X ./ r2;

figure;
quiver(X, Y, U, V);
title('Vortex in a Uniform Flow');
xlabel('X');
ylabel('Y');
axis equal;
