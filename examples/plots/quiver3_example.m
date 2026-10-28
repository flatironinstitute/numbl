% quiver3 - 3D vector field plot
% Unit normals drawn on a paraboloid surface.

[X, Y] = meshgrid(linspace(-1, 1, 7));
Z = X.^2 + Y.^2;
U = -2 * X;
V = -2 * Y;
W = ones(size(Z));
n = sqrt(U.^2 + V.^2 + W.^2);

figure;
surf(X, Y, Z);
hold on;
quiver3(X, Y, Z, U ./ n, V ./ n, W ./ n, 0.3);
hold off;
title('Surface Normals of a Paraboloid');
xlabel('X');
ylabel('Y');
zlabel('Z');
