% Test coordinate transform builtins: cart2sph, sph2cart, cart2pol, pol2cart

tol = 1e-10;

%% cart2sph - scalar
[az, el, r] = cart2sph(1, 0, 0);
assert(abs(az) < tol);
assert(abs(el) < tol);
assert(abs(r - 1) < tol);

[az, el, r] = cart2sph(0, 1, 0);
assert(abs(az - pi/2) < tol);
assert(abs(el) < tol);

[az, el, r] = cart2sph(0, 0, 1);
assert(abs(el - pi/2) < tol);
assert(abs(r - 1) < tol);

[az, el, r] = cart2sph(1, 1, 1);
assert(abs(az - pi/4) < tol);
assert(abs(r - sqrt(3)) < tol);

[az, el, r] = cart2sph(0, 0, 0);
assert(r == 0);

%% cart2sph - vector
x = [1, 0, 0]; y = [0, 1, 0]; z = [0, 0, 1];
[az, el, r] = cart2sph(x, y, z);
assert(abs(az(1)) < tol);
assert(abs(az(2) - pi/2) < tol);
assert(abs(el(3) - pi/2) < tol);

%% sph2cart - scalar
[x, y, z] = sph2cart(0, 0, 1);
assert(abs(x - 1) < tol);
assert(abs(y) < tol);
assert(abs(z) < tol);

[x, y, z] = sph2cart(pi/2, 0, 1);
assert(abs(x) < tol);
assert(abs(y - 1) < tol);
assert(abs(z) < tol);

[x, y, z] = sph2cart(0, pi/2, 1);
assert(abs(x) < tol);
assert(abs(y) < tol);
assert(abs(z - 1) < tol);

[x, y, z] = sph2cart(0, 0, 0);
assert(abs(x) < tol);
assert(abs(y) < tol);
assert(abs(z) < tol);

%% sph2cart - vector
az = [0, pi/2, 0]; el = [0, 0, pi/2]; r = [1, 1, 1];
[x, y, z] = sph2cart(az, el, r);
assert(abs(x(1) - 1) < tol);
assert(abs(y(2) - 1) < tol);
assert(abs(z(3) - 1) < tol);

%% round-trip: cart2sph -> sph2cart
x0 = 3; y0 = -4; z0 = 5;
[az, el, r] = cart2sph(x0, y0, z0);
[x1, y1, z1] = sph2cart(az, el, r);
assert(abs(x1 - x0) < tol);
assert(abs(y1 - y0) < tol);
assert(abs(z1 - z0) < tol);

%% round-trip with vectors
xv = [1, -2, 3, 0]; yv = [4, 0, -1, 5]; zv = [0, 3, -2, 1];
[azv, elv, rv] = cart2sph(xv, yv, zv);
[xr, yr, zr] = sph2cart(azv, elv, rv);
assert(max(abs(xr - xv)) < tol);
assert(max(abs(yr - yv)) < tol);
assert(max(abs(zr - zv)) < tol);

%% cart2pol - 2D scalar
[th, rho] = cart2pol(1, 0);
assert(abs(th) < tol);
assert(abs(rho - 1) < tol);

[th, rho] = cart2pol(0, 1);
assert(abs(th - pi/2) < tol);
assert(abs(rho - 1) < tol);

[th, rho] = cart2pol(1, 1);
assert(abs(th - pi/4) < tol);
assert(abs(rho - sqrt(2)) < tol);

[th, rho] = cart2pol(-1, 0);
assert(abs(abs(th) - pi) < tol);

%% cart2pol - 3D (cylindrical)
[th, rho, zout] = cart2pol(1, 0, 5);
assert(abs(th) < tol);
assert(abs(rho - 1) < tol);
assert(abs(zout - 5) < tol);

%% cart2pol - vector
x = [1, 0, -1]; y = [0, 1, 0];
[th, rho] = cart2pol(x, y);
assert(abs(th(1)) < tol);
assert(abs(th(2) - pi/2) < tol);
assert(abs(rho(1) - 1) < tol);
assert(abs(rho(2) - 1) < tol);

%% pol2cart - 2D scalar
[x, y] = pol2cart(0, 1);
assert(abs(x - 1) < tol);
assert(abs(y) < tol);

[x, y] = pol2cart(pi/2, 1);
assert(abs(x) < tol);
assert(abs(y - 1) < tol);

[x, y] = pol2cart(pi/4, sqrt(2));
assert(abs(x - 1) < tol);
assert(abs(y - 1) < tol);

%% pol2cart - 3D (cylindrical)
[x, y, zout] = pol2cart(0, 1, 7);
assert(abs(x - 1) < tol);
assert(abs(y) < tol);
assert(abs(zout - 7) < tol);

%% pol2cart - vector
th = [0, pi/2, pi]; rho = [1, 2, 3];
[x, y] = pol2cart(th, rho);
assert(abs(x(1) - 1) < tol);
assert(abs(y(2) - 2) < tol);
assert(abs(x(3) + 3) < tol);

%% round-trip: cart2pol -> pol2cart (2D)
x0 = 3; y0 = -4;
[th, rho] = cart2pol(x0, y0);
[x1, y1] = pol2cart(th, rho);
assert(abs(x1 - x0) < tol);
assert(abs(y1 - y0) < tol);

%% round-trip: cart2pol -> pol2cart (3D cylindrical)
x0 = -2; y0 = 5; z0 = 3;
[th, rho, zc] = cart2pol(x0, y0, z0);
[x1, y1, z1] = pol2cart(th, rho, zc);
assert(abs(x1 - x0) < tol);
assert(abs(y1 - y0) < tol);
assert(abs(z1 - z0) < tol);

%% round-trip with vectors (2D)
xv = [1, -2, 3, 0]; yv = [4, 0, -1, 5];
[thv, rhov] = cart2pol(xv, yv);
[xr, yr] = pol2cart(thv, rhov);
assert(max(abs(xr - xv)) < tol);
assert(max(abs(yr - yv)) < tol);

disp('SUCCESS');
