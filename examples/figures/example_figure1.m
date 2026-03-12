% MATLAB Script: Example Figure with Multiple Subplots

clear; clc; close all;

% Generate sample data
x = linspace(0, 2*pi, 200);
y1 = sin(x);
y2 = cos(x);
y3 = sin(2*x);
y4 = cos(2*x);

% Create figure
figure('Name','Example Multi-Plot Figure','NumberTitle','off');

% -------------------------
% Subplot 1
% -------------------------
subplot(2,2,1)
plot(x, y1, 'b-', 'LineWidth', 1.5)
grid on
title('sin(x)')
xlabel('x')
ylabel('Amplitude')
legend('sin(x)', 'Location', 'best')

% -------------------------
% Subplot 2
% -------------------------
subplot(2,2,2)
plot(x, y2, 'r-', 'LineWidth', 1.5)
grid on
title('cos(x)')
xlabel('x')
ylabel('Amplitude')
legend('cos(x)', 'Location', 'best')

% -------------------------
% Subplot 3
% -------------------------
subplot(2,2,3)
plot(x, y3, 'g-', 'LineWidth', 1.5)
grid on
title('sin(2x)')
xlabel('x')
ylabel('Amplitude')
legend('sin(2x)', 'Location', 'best')

% -------------------------
% Subplot 4
% -------------------------
subplot(2,2,4)
plot(x, y4, 'm-', 'LineWidth', 1.5)
grid on
title('cos(2x)')
xlabel('x')
ylabel('Amplitude')
legend('cos(2x)', 'Location', 'best')

% Add overall figure title
sgtitle('Trigonometric Functions Example')

% Improve spacing
set(gcf, 'Color', 'w')