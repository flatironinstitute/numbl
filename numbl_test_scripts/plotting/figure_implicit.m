% A draw before any explicit `figure` uses the implicit figure 1, so the next
% no-arg `figure` must create a new figure (handle 2) rather than colliding.
% `plot` is routed through the shared plot dispatch.

plot(1:3, 1:3);
h = figure;
assert(h == 2, 'figure after an implicit plot should be 2');

disp('SUCCESS')
