% inpolygon: query points inside or on the edge of a polygon.

%% Unit square: inside, outside, and on-edge membership
xv = [0 1 1 0];
yv = [0 0 1 1];
assert(inpolygon(0.5, 0.5, xv, yv) == true, 'center is inside');
assert(inpolygon(2, 0.5, xv, yv) == false, 'point outside');

[in, on] = inpolygon([0.5 0 0.5], [0.5 0 0], xv, yv);
assert(isequal(in, logical([1 1 1])), 'inside-or-on');
assert(isequal(on, logical([0 1 1])), 'corner and edge midpoint are on');

%% Output is logical and the same size as the query arrays
xq = [0.5 -1; 0.5 2];
yq = [0.5 0.5; 0.9 0.5];
inM = inpolygon(xq, yq, xv, yv);
assert(islogical(inM), 'logical output');
assert(isequal(size(inM), [2 2]), 'shape matches query');
assert(isequal(inM, logical([1 0; 1 0])), 'matrix membership');

%% Multiply connected: square with a square hole (NaN-separated, opposite
%% orientation). Points in the hole are outside.
xvh = [1 4 4 1 1 NaN 2 2 3 3 2];
yvh = [1 1 4 4 1 NaN 2 3 3 2 2];
assert(inpolygon(1.5, 2.5, xvh, yvh) == true, 'solid region is inside');
assert(inpolygon(2.5, 2.5, xvh, yvh) == false, 'hole is outside');
assert(inpolygon(5, 5, xvh, yvh) == false, 'far outside');

%% Self-intersecting pentagram — exact counts from the MATLAB docs:
%% 6 strictly inside, 2 on the edge, 4 outside.
xvp = [0.5;0.2;1.0;0;0.8;0.5];
yvp = [1.0;0.1;0.7;0.7;0.1;1];
xqp = [0.1;0.5;0.9;0.2;0.4;0.5;0.5;0.9;0.6;0.8;0.7;0.2];
yqp = [0.4;0.6;0.9;0.7;0.3;0.8;0.2;0.4;0.4;0.6;0.2;0.6];
[inp, onp] = inpolygon(xqp, yqp, xvp, yvp);
assert(sum(inp) == 8, 'pentagram: 8 inside or on');
assert(sum(onp) == 2, 'pentagram: 2 on edge');
assert(sum(~inp) == 4, 'pentagram: 4 outside');
assert(sum(inp & ~onp) == 6, 'pentagram: 6 strictly inside');

disp('SUCCESS');
