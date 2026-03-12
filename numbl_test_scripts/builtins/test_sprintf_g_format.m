% Test sprintf with %g format specifier
% %g should switch to scientific notation when exponent >= 6 or <= -5

% Normal values stay as decimal
assert(strcmp(sprintf('%g', 3.14), '3.14'));
assert(strcmp(sprintf('%g', 100), '100'));
assert(strcmp(sprintf('%g', 100000), '100000'));
assert(strcmp(sprintf('%g', 0.0001), '0.0001'));

% Large values switch to scientific notation
assert(strcmp(sprintf('%g', 1000000), '1e+06'));
assert(strcmp(sprintf('%g', 1e10), '1e+10'));
assert(strcmp(sprintf('%g', 1.23e8), '1.23e+08'));

% Small values switch to scientific notation
assert(strcmp(sprintf('%g', 0.00001), '1e-05'));
assert(strcmp(sprintf('%g', 1e-10), '1e-10'));
assert(strcmp(sprintf('%g', 1.23e-7), '1.23e-07'));

% Zero stays as zero
assert(strcmp(sprintf('%g', 0), '0'));

% Precision with %g
assert(strcmp(sprintf('%.3g', 3.14159), '3.14'));
assert(strcmp(sprintf('%.10g', pi), '3.141592654'));

disp('SUCCESS');
