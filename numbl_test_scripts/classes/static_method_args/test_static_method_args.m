% Test that arguments with 'end' indexing are properly passed to static methods
% (not wrapped in deferred lambdas meant for index resolution)

% First call a different static method
x = StaticArgTest.add(3, 4);
assert(x == 7);

% Call static method with end-indexing expression as argument
zz = [10; 20; 30; 40; 50];
T = StaticArgTest.process(zz(2:end), zz(1:end));
assert(T == 4);

% Call with vertcat expression containing end indexing
a = [0; 0.5; zeros(3, 1)];
T2 = StaticArgTest.process([2*a(1);a(2:end)], [2*a(1);a(2:end)]);
assert(T2 == 5);

disp('SUCCESS');
