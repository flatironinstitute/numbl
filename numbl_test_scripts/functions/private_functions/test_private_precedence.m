% Test that private function takes precedence over workspace function
result = compute(5);
assert(result == 105);  % private/compute.m returns x + 100, not x + 1
disp('SUCCESS')
