% Test operator-name builtins (calling operators as functions)

% --- mtimes: matrix multiplication (a * b) ---
assert(mtimes(3, 4) == 12);
assert(mtimes(2.5, 4) == 10);
A = [1 2; 3 4];
B = [5 6; 7 8];
C = mtimes(A, B);
assert(isequal(C, A * B));
assert(isequal(C, [19 22; 43 50]));

% --- mrdivide: matrix right division (a / b) ---
assert(mrdivide(10, 2) == 5);
assert(mrdivide(7, 2) == 3.5);

% --- mldivide: matrix left division (a \ b) ---
assert(mldivide(2, 10) == 5);
assert(mldivide(4, 20) == 5);

% --- mpower: matrix power (a ^ b) ---
assert(mpower(2, 3) == 8);
assert(mpower(3, 2) == 9);

% --- ldivide: element-wise left division (a .\ b) ---
assert(ldivide(2, 10) == 5);
assert(ldivide(4, 20) == 5);
v1 = [2 4];
v2 = [10 20];
assert(isequal(ldivide(v1, v2), [5 5]));

% --- uminus: unary minus (-a) ---
assert(uminus(5) == -5);
assert(uminus(-3) == 3);
assert(isequal(uminus([1 -2 3]), [-1 2 -3]));

% --- uplus: unary plus (+a) ---
assert(uplus(5) == 5);
assert(uplus(-3) == -3);
assert(isequal(uplus([1 -2 3]), [1 -2 3]));

% --- Verify consistency with operators ---
x = 6; y = 3;
assert(mtimes(x, y) == x * y);
assert(mrdivide(x, y) == x / y);
assert(mpower(x, y) == x ^ y);
assert(ldivide(x, y) == x .\ y);
assert(mldivide(x, y) == x \ y);
assert(uminus(x) == -x);
assert(uplus(x) == +x);

disp('SUCCESS');
