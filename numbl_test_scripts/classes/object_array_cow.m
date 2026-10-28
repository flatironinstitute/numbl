% Copy-on-write for arrays of value-class objects: modifying a copy must
% not change the original. Uses the ArrItem_ value class.

% element property assign through a copy
a = [ArrItem_(1) ArrItem_(2) ArrItem_(3)];
b = a;
b(1).v = 99;
assert(a(1).v == 1);
assert(b(1).v == 99);
assert(b(2).v == 2);

% element replacement through a copy
c = a;
c(2) = ArrItem_(7);
assert(a(2).v == 2);
assert(c(2).v == 7);

% growth through a copy
d = a;
d(4) = ArrItem_(4);
assert(numel(a) == 3);
assert(numel(d) == 4);

% object array held in a struct field
s.arr = a;
t = s;
t.arr(3).v = -1;
assert(s.arr(3).v == 3);
assert(t.arr(3).v == -1);

% object array held in a cell
q = {a};
r = q;
r{1}(2).v = 50;
assert(q{1}(2).v == 2);
assert(r{1}(2).v == 50);

% subarray from indexing is an independent copy
sub = a(2:3);
sub(1).v = 20;
assert(a(2).v == 2);
assert(sub(1).v == 20);
sub2 = sub;
sub2(2).v = 30;
assert(sub(2).v == 3);
assert(sub2(2).v == 30);

% copy inside a function does not affect the caller
e = a;
e2 = object_array_cow_helper(e);
assert(e(1).v == 1);
assert(e2(1).v == 1000);

disp('SUCCESS')

function x = object_array_cow_helper(x)
  x(1).v = 1000;
end
