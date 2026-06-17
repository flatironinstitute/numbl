% isosurface: extract a triangulated surface from a 3-D scalar volume via
% marching cubes (shared vertices by default).

%% Struct form matches the MATLAB doc example exactly (1693 verts, 3348 faces)
[x,y,z] = meshgrid(-3:0.25:3);
V = x.*exp(-x.^2 - y.^2 - z.^2);
s = isosurface(x,y,z,V,1e-4);
assert(size(s.vertices,2) == 3, 'vertices are N-by-3');
assert(size(s.faces,2) == 3, 'faces are M-by-3');
assert(size(s.vertices,1) == 1693, 'doc example vertex count');
assert(size(s.faces,1) == 3348, 'doc example face count');

%% [faces,verts] array form returns faces first, then vertices
[f, v] = isosurface(x,y,z,V,1e-4);
assert(isequal(size(v), size(s.vertices)), 'same vertices');
assert(isequal(size(f), size(s.faces)), 'same faces');
assert(min(f(:)) >= 1, 'faces are 1-based');

%% Sphere: vertices of isosurface(R,1) lie on the unit sphere
[gx,gy,gz] = meshgrid(-2:0.2:2);
R = sqrt(gx.^2 + gy.^2 + gz.^2);
sp = isosurface(gx,gy,gz,R,1.0);
rad = sqrt(sum(sp.vertices.^2, 2));
assert(max(abs(rad - 1.0)) < 0.15, 'vertices near the unit sphere');

%% Shared vertices form a closed manifold: every edge is in exactly 2 faces
fc = sp.faces;
E = sort([fc(:,[1 2]); fc(:,[2 3]); fc(:,[1 3])], 2);
[~,~,ic] = unique(E, 'rows');
assert(all(accumarray(ic,1) == 2), 'closed manifold (shared vertices)');

%% 'noshare' produces strictly more (duplicated) vertices
sn = isosurface(gx,gy,gz,R,1.0,'noshare');
assert(size(sn.vertices,1) > size(sp.vertices,1), 'noshare duplicates vertices');

%% Implicit coordinates (V, isovalue) and the distmesh signed-distance pattern
s2 = isosurface(R, 1.0);
assert(size(s2.vertices,1) == size(sp.vertices,1), 'implicit coords same count');

[ax,ay,az] = ndgrid(-1.2:0.15:1.2, -1.2:0.15:1.2, -1.2:0.15:1.2);
fd = sqrt(ax.^2 + ay.^2 + az.^2) - 1;   % signed distance, 0 = surface
pv = isosurface(ax, ay, az, fd, 0);
radn = sqrt(sum(pv.vertices.^2, 2));
assert(max(abs(radn - 1)) < 0.1, 'isosurface(fd,0) on the unit sphere (ndgrid)');

%% Colors: [faces,verts,colors] gives one interpolated value per vertex
[~, v3, c3] = isosurface(ax,ay,az,fd,0,ax);
assert(numel(c3) == size(v3,1), 'one color per vertex');

disp('SUCCESS');
