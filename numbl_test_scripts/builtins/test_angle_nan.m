% Test angle(NaN) should return NaN
% MATLAB: angle(NaN) = NaN, angle(Inf) = 0, angle(-Inf) = pi

assert(isnan(angle(NaN)), 'angle(NaN) should be NaN');
assert(angle(Inf) == 0, 'angle(Inf) should be 0');
assert(abs(angle(-Inf) - pi) < 1e-10, 'angle(-Inf) should be pi');
assert(angle(0) == 0, 'angle(0) should be 0');
assert(angle(5) == 0, 'angle(positive) should be 0');
assert(abs(angle(-5) - pi) < 1e-10, 'angle(negative) should be pi');

disp('SUCCESS');
