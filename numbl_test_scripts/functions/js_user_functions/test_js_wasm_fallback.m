% Test .js user function with wasm binding and JS fallback
result = wadd(3, 4);
assert(result == 7, 'Expected wadd(3, 4) == 7');

result2 = wadd(10.5, 20.3);
assert(abs(result2 - 30.8) < 1e-10, 'Expected wadd(10.5, 20.3) == 30.8');

disp('SUCCESS')
