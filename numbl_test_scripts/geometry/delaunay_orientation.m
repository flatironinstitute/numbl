% delaunay: 2-D triangles must be returned with a consistent
% counterclockwise (CCW) winding, i.e. positive signed area.
%
% MATLAB's delaunay guarantees CCW-oriented triangles; code that walks the
% mesh boundary or computes signed areas relies on this. Verified against
% MATLAB R2025b.

x = [0.12; 0.83; 0.45; 0.97; 0.31; 0.68; 0.05; 0.59; 0.24; 0.91; 0.40; 0.76];
y = [0.34; 0.11; 0.88; 0.52; 0.07; 0.95; 0.61; 0.29; 0.73; 0.40; 0.18; 0.66];

f = delaunay(x, y);

% Signed area of each triangle (positive => CCW).
ax = x(f(:, 1)); ay = y(f(:, 1));
bx = x(f(:, 2)); by = y(f(:, 2));
cx = x(f(:, 3)); cy = y(f(:, 3));
signedArea = 0.5 * ((bx - ax) .* (cy - ay) - (cx - ax) .* (by - ay));

assert(all(signedArea > 0), ...
    'delaunay returned triangles that are not consistently CCW-oriented');

% A small disk-like mesh (points on concentric rings) — same guarantee.
nt = 16;
px = 0; py = 0;
for ring = 1:4
    th = (0:nt-1) * (2*pi/nt) + 0.1*ring;
    px = [px, (ring/4) * cos(th)]; %#ok<AGROW>
    py = [py, (ring/4) * sin(th)]; %#ok<AGROW>
end
px = px(:); py = py(:);
f2 = delaunay(px, py);
a2 = 0.5 * ((px(f2(:,2)) - px(f2(:,1))) .* (py(f2(:,3)) - py(f2(:,1))) ...
          - (px(f2(:,3)) - px(f2(:,1))) .* (py(f2(:,2)) - py(f2(:,1))));
assert(all(a2 > 0), ...
    'delaunay (disk mesh) returned triangles that are not consistently CCW-oriented');

disp('SUCCESS')
