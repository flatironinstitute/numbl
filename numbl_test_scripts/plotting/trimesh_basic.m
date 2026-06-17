% trimesh: triangular mesh plot. Builds a patch from a connectivity matrix T
% and vertex coordinates, returning a patch handle.

% Two triangles sharing an edge (a unit square split along the diagonal).
x = [0; 1; 1; 0];
y = [0; 0; 1; 1];
T = [1 2 3; 1 3 4];

% Default colors (per the docs): light-gray faces, black edges.
h = trimesh(T, x, y);
assert(isequal(h.FaceColor, [0.85 0.85 0.85]), 'default FaceColor light gray');
assert(isequal(h.EdgeColor, [0 0 0]), 'default EdgeColor black');

% Explicit FaceColor/EdgeColor name-value pairs (the DistMesh idiom).
h2 = trimesh(T, x, y, 'FaceColor', [0.8 0.9 1], 'EdgeColor', 'k');
assert(isequal(h2.FaceColor, [0.8 0.9 1]), 'FaceColor name-value');
assert(isequal(h2.EdgeColor, [0 0 0]), 'EdgeColor "k" -> black');

% 3-D form with a z coordinate, plus a line-width property.
z = [0; 0; 1; 1];
h3 = trimesh(T, x, y, z, 'LineWidth', 2);
assert(h3.LineWidth == 2, 'LineWidth name-value');

% The DistMesh call shape: z = zeros, explicit colors as the 4th+ args.
trimesh(T, x, y, 0*x, 'facecolor', [0.8 0.9 1], 'edgecolor', 'k');

disp('SUCCESS');
