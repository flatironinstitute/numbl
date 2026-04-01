% plot3 - 3D line plot
% Demonstrates a 3D parametric curve (helix)

t = linspace(0, 6*pi, 500);
x = cos(t);
y = sin(t);
z = t / (2*pi);

figure;
plot3(x, y, z, 'b-');
title('Helix in 3D');
xlabel('X');
ylabel('Y');
zlabel('Z');
grid on;
