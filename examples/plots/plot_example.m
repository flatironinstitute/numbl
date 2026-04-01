% plot - Basic 2D line plot
% Demonstrates single and multiple line plots with styling

% Simple sine and cosine curves
t = linspace(0, 2*pi, 200);
y1 = sin(t);
y2 = cos(t);

figure;
plot(t, y1, 'b-', t, y2, 'r--');
title('Sine and Cosine Waves');
xlabel('Angle (radians)');
ylabel('Amplitude');
legend('sin(t)', 'cos(t)');
grid on;
