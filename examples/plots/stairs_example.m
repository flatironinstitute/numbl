% stairs - Stairstep graph
% Demonstrates a stairstep plot

x = 0:0.25:4;
y = sin(x);

figure;
stairs(x, y);
title('Stairstep Graph');
xlabel('x');
ylabel('sin(x)');
grid on;
