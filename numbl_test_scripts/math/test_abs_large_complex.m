% abs of a large-magnitude complex must not overflow (hypot semantics)
r1 = abs(1e200 + 1e200i);
if ~isfinite(r1)
    error('abs(1e200+1e200i) not finite: %g', r1);
end
expected1 = 1e200 * sqrt(2);
if abs(r1 / expected1 - 1) > 1e-12
    error('abs(1e200+1e200i)=%g, expected %g', r1, expected1);
end

r2 = abs(3e200 + 4e200i);
expected2 = 5e200;
if abs(r2 / expected2 - 1) > 1e-12
    error('abs(3e200+4e200i)=%g, expected %g', r2, expected2);
end

% Pure imaginary still works
r3 = abs(1e200i);
if r3 ~= 1e200
    error('abs(1e200i)=%g', r3);
end

disp('SUCCESS');
