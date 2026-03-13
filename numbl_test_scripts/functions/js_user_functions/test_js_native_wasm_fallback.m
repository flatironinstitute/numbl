% Test .js user function with native + wasm + JS fallback chain (L2 norm)
result = mynorm([3 4]);
assert(result == 5, 'Expected mynorm([3 4]) == 5');

result2 = mynorm([1 2 3 4 5 6 7 8 9 10]);
expected = sqrt(385);
assert(abs(result2 - expected) < 1e-10, 'Expected mynorm(1:10) == sqrt(385)');

result3 = mynorm([0 0 0]);
assert(result3 == 0, 'Expected mynorm([0 0 0]) == 0');

disp('SUCCESS')
