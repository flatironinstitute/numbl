% Test asin, acos, atan with complex input

% asin(2i) = 0 + 1.4436i = log(2+sqrt(5))*i
z1 = 2i;
r1 = asin(z1);
assert(abs(real(r1)) < 1e-10);
assert(abs(imag(r1) - log(2 + sqrt(5))) < 1e-10);

% acos(2) = 0 + 1.3170i = -log(2-sqrt(3))*i
r2 = acos(2);
assert(abs(real(r2)) < 1e-10);
assert(abs(imag(r2) - (-log(2 - sqrt(3)))) < 1e-10);

% atan(2i) = -pi/2 + i*log(3)/2
r3 = atan(2i);
assert(abs(real(r3) - (-pi/2)) < 1e-10);
assert(abs(imag(r3) - log(3)/2) < 1e-10);

% Complex tensors: asin applied to a tensor with complex elements
% asin(2i) = 0+1.4436i, asin(0.5) = pi/6 (real)
v = [2i; 0.5];
rv = asin(v);
assert(abs(real(rv(1))) < 1e-10);
assert(abs(imag(rv(1)) - log(2 + sqrt(5))) < 1e-10);
assert(abs(real(rv(2)) - pi/6) < 1e-10);
assert(abs(imag(rv(2))) < 1e-10);

disp('SUCCESS')
