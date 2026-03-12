% Test str2double edge cases

% Empty string returns NaN
assert(isnan(str2double('')));

% Whitespace-only string returns NaN
assert(isnan(str2double('   ')));

% Normal conversions still work
assert(str2double('3.14') == 3.14);
assert(str2double('42') == 42);
assert(str2double('-1.5e3') == -1500);
assert(isnan(str2double('abc')));

disp('SUCCESS');
