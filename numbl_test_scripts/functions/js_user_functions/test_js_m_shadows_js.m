% Test that .m workspace function shadows .js user function with same name
%!numbl:assert_jit c
result = myshadowed();
assert(result == 42, 'Expected .m to shadow .js, got %d', result);

disp('SUCCESS')
