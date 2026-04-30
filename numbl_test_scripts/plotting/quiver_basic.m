% Basic quiver smoke test: a swirling 2D vector field.

theta = linspace(0, 2*pi, 8);
r = linspace(0.5, 2, 4);
[T, R] = meshgrid(theta, r);
X = R .* cos(T);
Y = R .* sin(T);
U = -Y;
V = X;

quiver(X, Y, U, V);
title('Swirl');
axis equal;

% quiver(U, V) form, no auto-scaling
figure;
quiver(U, V, 0);

disp('SUCCESS');
