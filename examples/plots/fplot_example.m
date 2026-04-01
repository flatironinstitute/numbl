% fplot example - plot functions from handles
subplot(2, 2, 1);
fplot(@sin, [-2*pi 2*pi]);
title('sin(x)');
grid on;

subplot(2, 2, 2);
fplot(@(x) x.^2 - 3*x + 1, [-2 5], 'r');
title('x^2 - 3x + 1');
grid on;

subplot(2, 2, 3);
fplot(@cos, [-2*pi 2*pi], 'g--');
title('cos(x)');
grid on;

subplot(2, 2, 4);
fplot(@(t) sin(t), @(t) cos(t), [0 2*pi]);
title('Parametric circle');
grid on;
