% linspace with Inf/NaN bounds: MATLAB preserves endpoints exactly and
% samples midpoints in a way that doesn't produce spurious NaN.
r1 = linspace(-Inf, Inf, 3);
% MATLAB: [-Inf, 0, Inf]
if r1(1) ~= -Inf, error('linspace(-Inf,Inf,3)(1)=%g', r1(1)); end
if r1(2) ~= 0,    error('linspace(-Inf,Inf,3)(2)=%g', r1(2)); end
if r1(3) ~= Inf,  error('linspace(-Inf,Inf,3)(3)=%g', r1(3)); end

r2 = linspace(1, Inf, 5);
% MATLAB: [1, Inf, Inf, Inf, Inf]
if r2(1) ~= 1,   error('linspace(1,Inf,5)(1)=%g', r2(1)); end
if r2(5) ~= Inf, error('linspace(1,Inf,5)(5)=%g', r2(5)); end

r3 = linspace(NaN, 5, 5);
% MATLAB: end point preserved
if r3(5) ~= 5, error('linspace(NaN,5,5)(5)=%g', r3(5)); end

r4 = linspace(1, NaN, 5);
% MATLAB: start point preserved
if r4(1) ~= 1, error('linspace(1,NaN,5)(1)=%g', r4(1)); end

r5 = linspace(-Inf, 0, 3);
% MATLAB: [-Inf, NaN, 0]
if r5(1) ~= -Inf, error('linspace(-Inf,0,3)(1)=%g', r5(1)); end
if r5(3) ~= 0,    error('linspace(-Inf,0,3)(3)=%g', r5(3)); end

disp('SUCCESS');
