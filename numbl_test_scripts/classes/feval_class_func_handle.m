% Test that @ClassName inside a method of ClassName produces a valid function
% handle. This mimics chebfun's @chebtech2/get.m which returns @chebtech2
% from within a chebtech2 method.

% Create an instance, then call its method that returns @BaseTech
bt = BaseTech([0]);
h = bt.getTechHandle();

% func2str should preserve the class name
assert(strcmp(func2str(h), 'BaseTech'), ...
    sprintf('func2str should return ''BaseTech'', got ''%s''', func2str(h)));

% feval should call the constructor and return a class instance
obj = feval(h, [1 2 3]);
assert(isa(obj, 'BaseTech'), ...
    sprintf('feval should create BaseTech, got %s', class(obj)));
assert(isequal(obj.coeffs, [1 2 3]), 'Constructor args should be passed through');

disp('SUCCESS');
