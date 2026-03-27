% Test dec2hex and dec2bin with negative numbers
% MATLAB: uses two's complement representation for negatives

% dec2hex negative
assert(strcmp(dec2hex(-5), 'FB'), 'dec2hex(-5) should be FB');
assert(strcmp(dec2hex(-1), 'FF'), 'dec2hex(-1) should be FF');
assert(strcmp(dec2hex(-256), 'FF00'), 'dec2hex(-256) should be FF00');

% dec2bin negative
assert(strcmp(dec2bin(-5), '11111011'), 'dec2bin(-5) should be 11111011');
assert(strcmp(dec2bin(-1), '11111111'), 'dec2bin(-1) should be 11111111');
assert(strcmp(dec2bin(-128), '10000000'), 'dec2bin(-128) should be 10000000');

% positive values should still work
assert(strcmp(dec2hex(255), 'FF'), 'dec2hex(255) should be FF');
assert(strcmp(dec2bin(5), '101'), 'dec2bin(5) should be 101');

disp('SUCCESS');
