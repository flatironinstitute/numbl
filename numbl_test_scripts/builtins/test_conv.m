% Test conv and deconv builtins

% Basic convolution
a = [1 2 3];
b = [1 1];
c = conv(a, b);
assert(isequal(c, [1 3 5 3]));

% Convolution with 'same'
c2 = conv(a, b, 'same');
assert(isequal(c2, [3 5 3]));

% Convolution with 'valid'
c3 = conv(a, b, 'valid');
assert(isequal(c3, [3 5]));

% Full (default)
c4 = conv(a, b, 'full');
assert(isequal(c4, [1 3 5 3]));

% Convolve with longer kernel
a2 = [1 0 1];
b2 = [2 3];
c5 = conv(a2, b2);
assert(isequal(c5, [2 3 2 3]));

% Deconvolution
[q, r] = deconv([1 3 5 3], [1 1]);
assert(isequal(q, [1 2 3]));
assert(all(abs(r) < 1e-10));

% Scalar convolution
c6 = conv(3, 4);
assert(c6 == 12);

disp('SUCCESS');
