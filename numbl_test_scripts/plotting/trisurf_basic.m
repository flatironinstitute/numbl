% trisurf: triangular surface plot. Builds a patch from a connectivity matrix
% T and vertex coordinates, colored by z (or c), returning a patch handle.

% Two triangles sharing an edge (a unit square split along the diagonal).
x = [0; 1; 1; 0];
y = [0; 0; 1; 1];
z = [0; 1; 2; 3];
T = [1 2 3; 1 3 4];

% Default: flat faces colored by the per-vertex z values.
h = trisurf(T, x, y, z);
assert(strcmp(class(h), 'matlab.graphics.primitive.Patch'), 'returns a patch');
assert(strcmp(h.FaceColor, 'flat'), 'default FaceColor flat');
cd = h.FaceVertexCData;
assert(isequal(cd(:), z), 'CData defaults to z');
assert(h.EdgeAlpha == 1, 'default EdgeAlpha');

% Explicit per-vertex color data.
c = [5; 6; 7; 8];
h2 = trisurf(T, x, y, z, c);
cd = h2.FaceVertexCData;
assert(isequal(cd(:), c), 'per-vertex c');

% Row-vector coordinates and Name-Value pairs; lighting strengths are accepted.
h3 = trisurf(T, x', y', z', c, 'EdgeColor', 'black', 'EdgeAlpha', 0.1, ...
    'AmbientStrength', 0.6, 'DiffuseStrength', 0.4, 'SpecularStrength', 0.3);
assert(isequal(h3.EdgeColor, [0 0 0]), 'EdgeColor name-value');
assert(h3.EdgeAlpha == 0.1, 'EdgeAlpha name-value');

% Explicit FaceColor overrides the default.
h4 = trisurf(T, x, y, z, 'FaceColor', [1 0 0]);
assert(isequal(h4.FaceColor, [1 0 0]), 'FaceColor name-value');

% Handle properties can be set after creation.
h4.FaceAlpha = 0.5;
assert(h4.FaceAlpha == 0.5, 'set FaceAlpha');

% The fmm3dbie surfer plot call shape.
trisurf(T, x, y, z, c, 'EdgeColor', 'black', 'EdgeAlpha', 0.1);
view(3); shading interp; axis equal; grid on;

disp('SUCCESS');
