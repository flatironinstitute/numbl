v = [10, 20, 30, 40, 50];
sub = v(3:end);
assert(length(sub) == 3)
assert(sub(1) == 30)
assert(sub(3) == 50)
disp('SUCCESS')
