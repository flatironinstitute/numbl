% idivide's optional third argument selects the rounding mode.  numbl
% used to ignore it and always truncate toward zero ('fix'), which
% produced silently-wrong results for 'floor', 'ceil', 'round'.

% --- 'fix' (default): truncate toward zero ---
assert(double(idivide(int32(-7), int32(2))) == -3);
assert(double(idivide(int32(-7), int32(2), 'fix')) == -3);
assert(double(idivide(int32( 7), int32(2), 'fix')) ==  3);
assert(double(idivide(int32( 7), int32(-2), 'fix')) == -3);

% --- 'floor': round toward -Inf ---
assert(double(idivide(int32(-7), int32(2), 'floor')) == -4);
assert(double(idivide(int32( 7), int32(2), 'floor')) ==  3);
assert(double(idivide(int32( 7), int32(-2), 'floor')) == -4);
assert(double(idivide(int32(-7), int32(-2), 'floor')) ==  3);

% --- 'ceil': round toward +Inf ---
assert(double(idivide(int32(-7), int32(2), 'ceil')) == -3);
assert(double(idivide(int32( 7), int32(2), 'ceil')) ==  4);
assert(double(idivide(int32( 7), int32(-2), 'ceil')) == -3);
assert(double(idivide(int32(-7), int32(-2), 'ceil')) ==  4);

% --- 'round': round to nearest, ties away from zero ---
assert(double(idivide(int32( 5), int32(2), 'round')) ==  3);
assert(double(idivide(int32(-5), int32(2), 'round')) == -3);
assert(double(idivide(int32( 7), int32(2), 'round')) ==  4);
assert(double(idivide(int32(-7), int32(2), 'round')) == -4);
assert(double(idivide(int32( 4), int32(2), 'round')) ==  2);  % exact

% --- vector form respects mode ---
r = idivide(int32([-7 7 -5 5]), int32(2), 'floor');
assert(isequal(double(r), [-4 3 -3 2]));

disp('SUCCESS');
