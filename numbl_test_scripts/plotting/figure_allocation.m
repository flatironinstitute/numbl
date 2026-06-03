% figure handle allocation (MATLAB-style): a no-arg `figure` creates a NEW
% figure each call; `figure(n)` selects/creates n and advances the counter.

a = figure;
b = figure;
c = figure;
assert(a == 1 && b == 2 && c == 3, 'no-arg figure should create new figures');

figure(7);
d = figure;
assert(d == 8, 'figure(n) should advance the counter');

disp('SUCCESS')
