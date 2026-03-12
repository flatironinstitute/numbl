% Test logical operators

% Basic logical
assert(true & true);
assert(~(true & false));
assert(true | false);
assert(~(false | false));

% Short-circuit operators
assert(true && true);
assert(~(true && false));
assert(true || false);
assert(~(false || false));

% xor
assert(xor(true, false));
assert(~xor(true, true));
assert(~xor(false, false));

% Comparison operators
assert(3 > 2);
assert(2 < 3);
assert(3 >= 3);
assert(3 <= 4);
assert(3 == 3);
assert(3 ~= 4);

% not operator
assert(~false);
assert(~(~true));

% Logical with numbers
assert(logical(1));
assert(~logical(0));

disp('SUCCESS')
