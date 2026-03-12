% Test that .js user function shadows a builtin with the same name
% sign.js overrides the builtin sign to return 999
result = sign(5);
assert(result == 999, 'Expected .js to shadow builtin sign');

disp('SUCCESS')
