% Test cart2sph: Cartesian to spherical coordinate transform

% Scalar: point along positive x-axis
[az, el, r] = cart2sph(1, 0, 0);
assert(abs(az) < 1e-10, 'az should be 0 for point on +x axis');
assert(abs(el) < 1e-10, 'el should be 0 for point on +x axis');
assert(abs(r - 1) < 1e-10, 'r should be 1');

% Scalar: point along positive y-axis
[az, el, r] = cart2sph(0, 1, 0);
assert(abs(az - pi/2) < 1e-10, 'az should be pi/2 for point on +y axis');
assert(abs(el) < 1e-10, 'el should be 0');
assert(abs(r - 1) < 1e-10, 'r should be 1');

% Scalar: point along positive z-axis
[az, el, r] = cart2sph(0, 0, 1);
assert(abs(el - pi/2) < 1e-10, 'el should be pi/2 for point on +z axis');
assert(abs(r - 1) < 1e-10, 'r should be 1');

% Scalar: point along negative x-axis
[az, el, r] = cart2sph(-1, 0, 0);
assert(abs(abs(az) - pi) < 1e-10, 'az should be pi or -pi for point on -x axis');

% Scalar: general point
[az, el, r] = cart2sph(1, 1, 1);
assert(abs(az - pi/4) < 1e-10, 'az should be pi/4');
assert(abs(r - sqrt(3)) < 1e-10, 'r should be sqrt(3)');

% Vector inputs
x = [1, 0, 0, 1];
y = [0, 1, 0, 1];
z = [0, 0, 1, 1];
[az, el, r] = cart2sph(x, y, z);
assert(abs(az(1)) < 1e-10);
assert(abs(az(2) - pi/2) < 1e-10);
assert(abs(el(3) - pi/2) < 1e-10);
assert(abs(r(4) - sqrt(3)) < 1e-10);

% Origin
[az, el, r] = cart2sph(0, 0, 0);
assert(r == 0, 'r should be 0 at origin');

% Single output returns azimuth only
az_only = cart2sph(0, 1, 0);
assert(abs(az_only - pi/2) < 1e-10, 'single output should be azimuth');

disp('SUCCESS');
