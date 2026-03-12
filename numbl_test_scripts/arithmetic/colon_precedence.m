% Test that colon (:) has higher precedence than comparison operators (==, <, etc.)
% In MATLAB, a == 1:3 is parsed as a == (1:3), not (a == 1):3

% Basic: colon binds tighter than ==
a = [1 2 3];
result = a == 1:3;
assert(isequal(result, [true true true]));

% Colon binds tighter than ~=
result2 = a ~= 1:3;
assert(isequal(result2, [false false false]));

% Colon binds tighter than <
result3 = a < 1:3;
assert(isequal(result3, [false false false]));

% Colon binds tighter than <=
result4 = a <= 1:3;
assert(isequal(result4, [true true true]));

% Colon binds tighter than >
result5 = a > 1:3;
assert(isequal(result5, [false false false]));

% Colon binds tighter than >=
result6 = a >= 1:3;
assert(isequal(result6, [true true true]));

% Three-part range with comparison
b = [2 4 6];
result7 = b == 2:2:6;
assert(isequal(result7, [true true true]));

disp('SUCCESS');
