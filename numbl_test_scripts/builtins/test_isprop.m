% isprop(obj, name): logical 1 where name is a property of class object obj.
% The result has the same size as obj. Methods are not properties. Structs
% (even with that field name), numeric, char and other built-in types always
% return false.

o = IsPropClass_();

% Public property -> logical scalar true.
r = isprop(o, 'width');
assert(islogical(r));
assert(isscalar(r));
assert(r);

% Private property is still a property -> true.
assert(isprop(o, 'secret'));

% A method is not a property.
assert(~isprop(o, 'area'));

% Non-existent name -> false.
assert(~isprop(o, 'nope'));

% String-scalar name argument works too.
assert(isprop(o, "height"));

% Object array -> logical array matching the object's size.
arr = [IsPropClass_(), IsPropClass_(), IsPropClass_()];
ra = isprop(arr, 'width');
assert(isequal(size(ra), [1 3]));
assert(all(ra(:)));
rb = isprop(arr, 'nope');
assert(isequal(size(rb), [1 3]));
assert(~any(rb(:)));

% Struct is NEVER an object, even when it has that field.
s.width = 1; s.height = 2;
assert(~isprop(s, 'width'));
assert(~isprop(s, 'nope'));

% Struct array -> false array matching size.
sa(1).width = 1; sa(2).width = 2;
ssz = isprop(sa, 'width');
assert(isequal(size(ssz), [1 2]));
assert(~any(ssz(:)));

% Numeric / char inputs -> false, with size matching the input.
assert(~isprop(5, 'x'));
n3 = isprop([1 2 3], 'x');
assert(isequal(size(n3), [1 3]));
assert(~any(n3(:)));
ch = isprop('hi', 'x');           % 'hi' is a 1x2 char row vector
assert(isequal(size(ch), [1 2]));
assert(~any(ch(:)));

% An invalid name argument (cell array) is not a single property name -> false.
assert(~isprop(o, {'width', 'height'}));

disp('SUCCESS')
