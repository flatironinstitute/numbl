% main.m — entry-point demo for this numbl project.
% Click "Run" above to execute it in your browser.

x = linspace(0, 2*pi, 200);
y = sin(x);

% Call a helper function defined in stats.m (same folder).
fprintf('mean of sin over [0, 2pi]: %.4f\n', stats(y, 'mean'));
fprintf('rms  of sin over [0, 2pi]: %.4f\n', stats(y, 'rms'));

plot(x, y, x, cos(x));
title('sin and cos');
legend('sin', 'cos');
xlabel('x');
grid on;
