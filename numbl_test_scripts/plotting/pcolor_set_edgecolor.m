% Test set(h,'EdgeColor','none') on pcolor handle

C = magic(5);
h = pcolor(C);
set(h, 'EdgeColor', 'none');

% Also test with X, Y
[X, Y] = meshgrid(1:4, 1:3);
Z = X + Y;
h2 = pcolor(X, Y, Z);
set(h2, 'EdgeColor', 'none');

disp('SUCCESS');
