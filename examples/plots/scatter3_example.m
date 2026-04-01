% scatter3 - Basic 3D scatter plot
% Demonstrates a 3D scatter plot with random data

x = randn(1, 100);
y = randn(1, 100);
z = x .* y + 0.3 * randn(1, 100);

figure;
scatter3(x, y, z, 'filled');
title('3D Scatter Plot');
xlabel('X');
ylabel('Y');
zlabel('Z');
