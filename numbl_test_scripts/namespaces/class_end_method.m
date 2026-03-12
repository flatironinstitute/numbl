% Test class with a method named 'end'

c = Counter(42);
assert(c.value == 42);

% Call the 'end' method
result = c.end(1, 1);
assert(result == 142);

disp('SUCCESS')
