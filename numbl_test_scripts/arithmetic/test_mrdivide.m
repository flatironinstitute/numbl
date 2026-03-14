% Matrix right division (mrdivide, /) tests

% scalar / scalar
assert(6 / 2 == 3)
assert(3 / 2 == 1.5)

% matrix / scalar (element-wise)
R = [1 2; 3 4] / 2;
assert(R(1,1) == 0.5)
assert(R(1,2) == 1)
assert(R(2,1) == 1.5)
assert(R(2,2) == 2)

% row vector / matrix (matrix right division)
R = [6 3] / [1 2; 3 4];
assert(abs(R(1) - (-7.5)) < 1e-10)
assert(abs(R(2) - 4.5) < 1e-10)

% matrix / matrix (matrix right division)
R = [1 2; 3 4] / [1 0; 0 2];
assert(R(1,1) == 1)
assert(R(1,2) == 1)
assert(R(2,1) == 3)
assert(R(2,2) == 2)

% scalar / matrix should error (dimensions don't agree)
threw = false;
try
    x = 3 / [1 2; 3 4];
catch
    threw = true;
end
assert(threw, 'scalar / matrix should error')

disp('SUCCESS')
