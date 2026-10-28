% Function handles to the true/false builtins (keyword-like names)
f = @true;
assert(islogical(f()) && f());
assert(strcmp(func2str(f), 'true'));
g = @false;
assert(isequal(g(2, 3), false(2, 3)));
assert(isequal(size(g(2, 3)), [2 3]));
c = {@true, @false};
assert(c{1}() && ~c{2}());
assert(isequal(cellfun(@(h) h(), c), [true false]));
h = @true;
assert(isequal(h(2), true(2)));
s = func2str(@false);
assert(strcmp(s, 'false'));
disp('SUCCESS');
