% startsWith / endsWith builtins

% --- startsWith basic ---
assert(startsWith("hello world", "hello"));
assert(~startsWith("hello world", "world"));
assert(startsWith('hello world', 'hello'));
assert(~startsWith('hello world', 'world'));

% --- endsWith basic ---
assert(endsWith("hello world", "world"));
assert(~endsWith("hello world", "hello"));
assert(endsWith('hello world', 'world'));
assert(~endsWith('hello world', 'hello'));

% --- cell array of patterns ---
assert(startsWith("hello", {'he', 'xyz'}));
assert(~startsWith("hello", {'xyz', 'abc'}));
assert(endsWith("hello", {'xyz', 'lo'}));
assert(~endsWith("hello", {'xyz', 'abc'}));

% --- IgnoreCase ---
assert(startsWith("Hello", "hello", 'IgnoreCase', true));
assert(~startsWith("Hello", "hello", 'IgnoreCase', false));
assert(endsWith("World", "world", 'IgnoreCase', true));
assert(~endsWith("World", "world", 'IgnoreCase', false));

% --- IgnoreCase with cell patterns ---
assert(startsWith("Hello", {'HE', 'xyz'}, 'IgnoreCase', true));
assert(endsWith("World", {'xyz', 'RLD'}, 'IgnoreCase', true));

% --- empty pattern ---
assert(startsWith("hello", ""));
assert(endsWith("hello", ""));

disp('SUCCESS');
