% Test log2 and log10 with complex input
% log2(z) = log(z) / log(2)
% log10(z) = log(z) / log(10)

% log2(2+2i)
% log(2+2i) = log(2*sqrt(2)) + i*pi/4
% / log(2) = 3/2 + i*pi/(4*log(2))
z = 2 + 2i;
r = log2(z);
expected_re = 3/2;
expected_im = pi / (4 * log(2));
assert(abs(real(r) - expected_re) < 1e-10);
assert(abs(imag(r) - expected_im) < 1e-10);

% log10(10i)
% log(10i) = log(10) + i*pi/2
% / log(10) = 1 + i*pi/(2*log(10))
z2 = 10i;
r2 = log10(z2);
assert(abs(real(r2) - 1) < 1e-10);
assert(abs(imag(r2) - pi/(2*log(10))) < 1e-10);

% Complex tensor: log2 on a tensor with complex elements
v = [2+2i; 10i];
rv = log2(v);
assert(abs(real(rv(1)) - 3/2) < 1e-10);
assert(abs(imag(rv(1)) - pi/(4*log(2))) < 1e-10);

disp('SUCCESS')
