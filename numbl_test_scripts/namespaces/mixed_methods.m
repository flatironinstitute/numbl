% Test class with mixed method signatures and function definitions
% The signatures (without bodies) are declarations for methods defined in
% separate files. Here we test that the function definitions parse and work.

y = MixedMethods_.func2(5);
assert(y == 6);

disp('SUCCESS')
