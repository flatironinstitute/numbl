% Test that inherited methods work via both dot syntax and function-call syntax.
% BaseAnimal_ has a 'describe' method. Dog_ overrides it.
% We add a new parent class with a method the child does NOT override.

% ParentWithMethod_ has greet() method; ChildNoOverride_ inherits it.
p = ParentWithMethod_(42);

% Test 1: dot syntax on parent instance
assert(p.greet() == 42);

% Test 2: function-call syntax on parent instance
assert(greet(p) == 42);

% Child that does NOT override greet()
c = ChildNoOverride_(42, 'hello');

% Test 3: dot syntax on child instance -> inherited method
assert(c.greet() == 42);

% Test 4: function-call syntax on child instance -> inherited method
assert(greet(c) == 42);

% Test 5: child's own method works
assert(strcmp(c.childOnly(), 'hello'));

disp('SUCCESS')
