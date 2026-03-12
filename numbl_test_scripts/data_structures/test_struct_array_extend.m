% Test: extending a struct array by assigning a struct to a new index
base(1).a = 100;
base(2).a = 200;
s2.a = 300;
base(3) = s2;
assert(numel(base) == 3);
assert(base(3).a == 300);

disp('SUCCESS');
