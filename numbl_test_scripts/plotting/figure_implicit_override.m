% Same implicit-figure rule for a draw handled OUTSIDE the shared dispatch
% (`line` has a custom runtime override): it must still register figure 1 so a
% following no-arg `figure` allocates handle 2.

line([0 1], [0 1]);
h = figure;
assert(h == 2, 'figure after an implicit line should be 2');

disp('SUCCESS')
