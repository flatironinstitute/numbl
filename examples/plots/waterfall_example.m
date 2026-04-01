% waterfall - Waterfall plot
% Demonstrates a waterfall plot of decaying ripples

x = linspace(-5, 5, 60);
y = linspace(0, 3, 20);
[X, Y] = meshgrid(x, y);
Z = exp(-Y) .* sin(X.^2);

figure;
waterfall(X, Y, Z);
title('Decaying Ripples');
xlabel('X');
ylabel('Y');
zlabel('Z');
