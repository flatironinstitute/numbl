% imagesc: XData/YData are pixel centers, so the axis limits are the pixel
% edges (tight); NaN entries are allowed.

imagesc(magic(4));
assert(isequal(axis, [0.5 4.5 0.5 4.5]));

clf;
imagesc([1 2 3]);
assert(isequal(axis, [0.5 3.5 0.5 1.5]));

clf;
imagesc([10 20], [5 8], rand(4, 3));
assert(isequal(axis, [7.5 22.5 4.5 8.5]));

clf;
Z = magic(6);
Z(2:4, 2:4) = NaN;
imagesc(Z);
assert(isequal(axis, [0.5 6.5 0.5 6.5]));

disp('SUCCESS');
