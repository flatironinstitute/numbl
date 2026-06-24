% `ClassName(x)` is always construction, even when an argument is an instance of
% that class and the class inherits a method whose name equals the class name
% (a converter). ConvLeaf_ inherits a method `ConvLeaf_` from ConvBase_, yet
% ConvLeaf_(existingLeaf) must build a new ConvLeaf_ via the constructor (which
% copies .data), not dispatch to the inherited same-named method.

x = ConvLeaf_();
x.data = 42;

y = ConvLeaf_(x);
assert(strcmp(class(y), 'ConvLeaf_'));
assert(y.data == 42, 'ClassName(instance) must construct via the constructor');

% Constructing from a matrix-free default still works.
z = ConvLeaf_();
assert(z.data == 0);

disp('SUCCESS')
