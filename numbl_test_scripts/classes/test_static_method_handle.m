% Test @ClassName.method syntax for static method function handles

% Get a handle to a static method
h = @StaticMethodClass.addOne;

% Call via handle
r = h(5);
assert(r == 6, 'static method handle call failed');

% Use in cellfun/arrayfun-like pattern
r2 = h(10);
assert(r2 == 11, 'static method handle second call failed');

disp('SUCCESS');
