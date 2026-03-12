% Test that .m workspace function shadows .js user function with same name
result = myshadowed();
assert(result == 42, 'Expected .m to shadow .js, got %d', result);

disp('SUCCESS')
