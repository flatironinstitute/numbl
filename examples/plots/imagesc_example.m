% imagesc - Image with scaled pixel values
% Demonstrates a heatmap-style image of a 2D function

x = linspace(-pi, pi, 80);
y = linspace(-pi, pi, 80);
[X, Y] = meshgrid(x, y);
Z = sin(X) .* cos(Y);

figure;
imagesc(x, y, Z);
title('sin(x) * cos(y)');
xlabel('X');
ylabel('Y');
colorbar;
