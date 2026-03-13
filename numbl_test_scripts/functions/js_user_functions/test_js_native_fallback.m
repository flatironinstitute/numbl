% Test .js user function with native directive and JS fallback
result = mydot([1 2 3], [4 5 6]);
assert(result == 32, 'Expected mydot([1 2 3], [4 5 6]) == 32');

result2 = mydot([1.5 2.5], [3.0 4.0]);
assert(abs(result2 - 14.5) < 1e-10, 'Expected mydot([1.5 2.5], [3.0 4.0]) == 14.5');

disp('SUCCESS')
