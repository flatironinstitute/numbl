% Test that cell() indexing with a matrix index preserves shape
c = {10; 20; 30; 40; 50; 60};
idx = [1 3; 2 4];
r = c(idx);
assert(iscell(r));
assert(size(r, 1) == 2);
assert(size(r, 2) == 2);
assert(r{1,1} == 10);
assert(r{2,1} == 20);
assert(r{1,2} == 30);
assert(r{2,2} == 40);

fprintf('SUCCESS\n');
