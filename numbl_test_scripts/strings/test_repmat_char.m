%% repmat with char - basic repeat
s = repmat(':', 1, 3);
assert(isequal(s, ':::'));
assert(isequal(class(s), 'char'));

%% repmat with char - zero columns gives empty char
s2 = repmat(':', 1, 0);
assert(isempty(s2));

%% repmat with multi-char string
s3 = repmat('ab', 1, 3);
assert(isequal(s3, 'ababab'));

%% repmat with char - single repeat
s4 = repmat('hello', 1, 1);
assert(isequal(s4, 'hello'));

disp('SUCCESS')
