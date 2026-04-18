% Test sqrt of negative real has exact zero real part (MATLAB-compatible)
x = sqrt(-4);
if real(x) ~= 0, error('sqrt(-4) real part: %g (expected 0)', real(x)); end
if imag(x) ~= 2, error('sqrt(-4) imag part: %g (expected 2)', imag(x)); end

y = sqrt(-1);
if real(y) ~= 0, error('sqrt(-1) real part: %g (expected 0)', real(y)); end
if imag(y) ~= 1, error('sqrt(-1) imag part: %g (expected 1)', imag(y)); end

% Tensor element-wise
v = sqrt([-4, -1, 4]);
if real(v(1)) ~= 0, error('sqrt([-4 -1 4])(1) real: %g', real(v(1))); end
if imag(v(1)) ~= 2, error('sqrt([-4 -1 4])(1) imag: %g', imag(v(1))); end
if real(v(2)) ~= 0, error('sqrt([-4 -1 4])(2) real: %g', real(v(2))); end
if imag(v(2)) ~= 1, error('sqrt([-4 -1 4])(2) imag: %g', imag(v(2))); end
if real(v(3)) ~= 2, error('sqrt([-4 -1 4])(3) real: %g', real(v(3))); end
if imag(v(3)) ~= 0, error('sqrt([-4 -1 4])(3) imag: %g', imag(v(3))); end

disp('SUCCESS');
