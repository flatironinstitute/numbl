% Test integer type constructors and idivide.  numbl represents all
% numeric data as doubles, so these tests check values, not the
% underlying MATLAB class — use isequal/== rather than class().

% --- int64 rounds and saturates ---
assert(int64(3.7) == 4, 'int64 round up');
assert(int64(3.4) == 3, 'int64 round down');
assert(int64(-3.7) == -4, 'int64 round away from zero neg');
assert(int64(0) == 0, 'int64 zero');

% --- int8 saturates ---
assert(int8(200) == 127, 'int8 saturate high');
assert(int8(-200) == -128, 'int8 saturate low');

% --- uint8 clamps negative to 0 ---
assert(uint8(-5) == 0, 'uint8 negative');
assert(uint8(300) == 255, 'uint8 high');

% --- vector conversion ---
v = int32([1.4 2.6 -3.2 100]);
assert(isequal(double(v), [1 3 -3 100]), 'int32 vector');

% --- idivide default (fix toward zero) ---
assert(double(idivide(int64(7), int64(2))) == 3, '7/2 fix');
assert(double(idivide(int64(-7), int64(2))) == -3, '-7/2 fix');
assert(double(idivide(int64(7), int64(-2))) == -3, '7/-2 fix');
assert(double(idivide(int64(-7), int64(-2))) == 3, '-7/-2 fix');

% --- chunkie/FLAM usage: bucketize indices ---
idx = [0 1 2 3 4 5 6 7];
opdim = 3;
bucket = double(idivide(int64(idx), int64(opdim))) + 1;
assert(isequal(bucket, [1 1 1 2 2 2 3 3]), 'bucketize');

% --- vector idivide ---
a = int64([10 20 30]);
b = int64([3 4 7]);
r = idivide(a, b);
assert(isequal(double(r), [3 5 4]), 'vec idivide');

disp('SUCCESS');
