% Test sprintf %e/%E exponent zero-padding
% MATLAB always pads exponents to at least 2 digits

assert(strcmp(sprintf('%e', 1000), '1.000000e+03'), 'sprintf %e 1000');
assert(strcmp(sprintf('%e', 0.001), '1.000000e-03'), 'sprintf %e 0.001');
assert(strcmp(sprintf('%E', 1000), '1.000000E+03'), 'sprintf %E 1000');
assert(strcmp(sprintf('%.2e', 1000), '1.00e+03'), 'sprintf %.2e 1000');
assert(strcmp(sprintf('%e', 1), '1.000000e+00'), 'sprintf %e 1');
assert(strcmp(sprintf('%e', 1e10), '1.000000e+10'), 'sprintf %e 1e10');
assert(strcmp(sprintf('%e', 1e-5), '1.000000e-05'), 'sprintf %e 1e-5');
assert(strcmp(sprintf('%e', -0.5), '-5.000000e-01'), 'sprintf %e negative');

disp('SUCCESS');
