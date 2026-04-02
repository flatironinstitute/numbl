% Comprehensive tests for the ans variable

function result = helper_returns_42()
    result = 42;
end

function result = helper_returns_vector()
    result = [true, false, true];
end

% Test 1: bare number sets ans
5;
assert(ans == 5, 'bare number should set ans');

% Test 2: suppressed expression overwrites ans
7;
assert(ans == 7, 'suppressed expression should overwrite ans');

% Test 3: function call without LHS sets ans
sin(0.5);
assert(abs(ans - sin(0.5)) < 1e-10, 'builtin call without LHS should set ans');

% Test 4: assignment does NOT change ans
3;
x = 99;
assert(ans == 3, 'assignment should not change ans');

% Test 5: char expression sets ans
'hello';
assert(strcmp(ans, 'hello'), 'char expression should set ans');

% Test 6: matrix expression sets ans
[1 2; 3 4];
assert(isequal(ans, [1 2; 3 4]), 'matrix expression should set ans');

% Test 7: logical expression sets ans
true;
assert(ans == true, 'logical expression should set ans');

% Test 8: struct expression sets ans
struct('a', 1);
assert(ans.a == 1, 'struct expression should set ans');

% Test 9: cell array expression sets ans
{1, 2, 3};
assert(isequal(ans, {1, 2, 3}), 'cell expression should set ans');

% Test 10: successive expressions — ans is last value
1;
2;
assert(ans == 2, 'ans should be the most recent expression value');

% Test 11: bare local function call sets ans
helper_returns_42;
assert(ans == 42, 'bare function call (no parens) should set ans');

% Test 12: local function call with parens sets ans
0;
helper_returns_42();
assert(ans == 42, 'function call with parens should set ans');

% Test 13: feval without LHS sets ans
0;
feval('helper_returns_42');
assert(ans == 42, 'feval without LHS should set ans');

% Test 14: feval returning vector sets ans to full vector
0;
feval('helper_returns_vector');
assert(isequal(ans, [true, false, true]), 'feval should set ans to full return value');

% Test 15: ans inside if block
0;
if true
    99;
end
assert(ans == 99, 'ans should be set inside if block');

% Test 16: ans inside for loop
0;
for i = 1:3
    sin(i);
end
assert(abs(ans - sin(3)) < 1e-10, 'ans should reflect last expression in loop');

% Test 17: ans can be used in expressions
5;
x = ans + 1;
assert(x == 6, 'ans should be usable in expressions');

% Test 18: ans in arithmetic
10;
y = ans * 2 + ans;
assert(y == 30, 'ans should work in compound arithmetic');

disp('SUCCESS');
