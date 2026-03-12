% Test: negative base with fractional exponent should return complex
% Bug: numbl returns NaN instead of complex result

% (-1)^0.5 should equal i
r1 = (-1)^0.5;
assert(abs(real(r1)) < 1e-10);
assert(abs(imag(r1) - 1) < 1e-10);

% (-4)^0.5 should equal 2i
r2 = (-4)^0.5;
assert(abs(real(r2)) < 1e-10);
assert(abs(imag(r2) - 2) < 1e-10);

% (-8)^(1/3) should be complex (principal root)
r3 = (-8)^(1/3);
% MATLAB: 1.0000 + 1.7321i
assert(abs(real(r3) - 1) < 1e-4);
assert(abs(imag(r3) - sqrt(3)) < 1e-4);

% (-2)^1.5 should be complex
r4 = (-2)^1.5;
% MATLAB: 0.0000 - 2.8284i
assert(abs(real(r4)) < 1e-4);
assert(abs(imag(r4) - (-2*sqrt(2))) < 1e-4);

% (-1)^(1/3) should be complex (principal root)
r5 = (-1)^(1/3);
% MATLAB: 0.5000 + 0.8660i
assert(abs(real(r5) - 0.5) < 1e-4);
assert(abs(imag(r5) - sqrt(3)/2) < 1e-4);

% Negative base with integer exponent should still work (real result)
assert((-2)^2 == 4);
assert((-2)^3 == -8);
assert((-3)^0 == 1);

disp('SUCCESS');
