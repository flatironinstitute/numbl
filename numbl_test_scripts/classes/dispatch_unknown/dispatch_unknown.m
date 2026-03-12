% Test: local function precedence with unknown types
% Even when the compiler can't infer types, local functions still take precedence.
%
% OffsetterD_ class has method "calc(obj, x)" that returns x + obj.Offset
% Local function "calc(~, ~)" always returns -999
%
% The X = 0; X = OffsetterD_(10) trick makes X have unknown type in the IR.
% Even so, calc(X, 5) should still call the local function (not the class method).

% Test 1: unknown type, runtime value is class instance
X = 0;
X = OffsetterD_(10);
r1 = calc(X, 5);
assert(r1 == -999);  % local function takes precedence regardless of type

% Test 2: unknown type, runtime value is number
Y = OffsetterD_(10);
Y = 7;
r2 = calc(Y, 3);
assert(r2 == -999);  % local function takes precedence

% Test 3: dot syntax still works with unknown type
Z = 0;
Z = OffsetterD_(20);
r3 = Z.calc(3);
assert(r3 == 23);  % class method: 3 + 20

disp('SUCCESS')

function r = calc(~, ~)
  r = -999;
end
