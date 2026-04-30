% Smoke test for tiledlayout / nexttile.

x = linspace(0, 2*pi, 50);

% Fixed 2x2 grid
figure;
tiledlayout(2, 2);
nexttile;
plot(x, sin(x));
title('sin');
nexttile;
plot(x, cos(x));
title('cos');
nexttile;
plot(x, sin(2*x));
title('sin(2x)');
nexttile;
plot(x, cos(2*x));
title('cos(2x)');

% Flow layout (no args)
figure;
tiledlayout;
nexttile; plot(x, sin(x));
nexttile; plot(x, cos(x));
nexttile; plot(x, x);

% Vertical and horizontal arrangements
figure;
tiledlayout('vertical');
nexttile; plot(x, sin(x));
nexttile; plot(x, cos(x));

figure;
tiledlayout('horizontal');
nexttile; plot(x, x);
nexttile; plot(x, x.^2);

% Skipping to a specific tile
figure;
tiledlayout(2, 3);
nexttile(1); plot(x, sin(x));
nexttile(5); plot(x, cos(x));

% Name-value options should be tolerated
figure;
tiledlayout(1, 2, 'TileSpacing', 'compact', 'Padding', 'tight');
nexttile; plot(x, sin(x));
nexttile; plot(x, cos(x));

disp('SUCCESS');
