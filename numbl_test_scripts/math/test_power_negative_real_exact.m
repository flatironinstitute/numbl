% (-x)^0.5 / power / mpower must produce exact 0 real part, matching sqrt
a = power(-4, 0.5);
if real(a) ~= 0, error('power(-4,0.5) re=%.17g, expected 0', real(a)); end
if imag(a) ~= 2, error('power(-4,0.5) im=%.17g, expected 2', imag(a)); end

b = (-4).^0.5;
if real(b) ~= 0, error('(-4).^0.5 re=%.17g, expected 0', real(b)); end
if imag(b) ~= 2, error('(-4).^0.5 im=%.17g, expected 2', imag(b)); end

c = mpower(-4, 0.5);
if real(c) ~= 0, error('mpower(-4,0.5) re=%.17g, expected 0', real(c)); end
if imag(c) ~= 2, error('mpower(-4,0.5) im=%.17g, expected 2', imag(c)); end

d = (-4)^0.5;
if real(d) ~= 0, error('(-4)^0.5 re=%.17g, expected 0', real(d)); end
if imag(d) ~= 2, error('(-4)^0.5 im=%.17g, expected 2', imag(d)); end

% tensor element-wise
v = [-4, -1, 4].^0.5;
if real(v(1)) ~= 0 || imag(v(1)) ~= 2, error('v(1) wrong: %g+%gi', real(v(1)), imag(v(1))); end
if real(v(2)) ~= 0 || imag(v(2)) ~= 1, error('v(2) wrong: %g+%gi', real(v(2)), imag(v(2))); end
if real(v(3)) ~= 2 || imag(v(3)) ~= 0, error('v(3) wrong: %g+%gi', real(v(3)), imag(v(3))); end

disp('SUCCESS');
