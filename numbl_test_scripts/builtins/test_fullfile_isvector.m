% fullfile: empty parts dropped, separators collapsed (except leading)
assert(strcmp(fullfile('a/', '/b', 'c'), 'a/b/c'));
assert(strcmp(fullfile('', 'a'), 'a'));
assert(strcmp(fullfile('a', ''), 'a'));
assert(strcmp(fullfile('a/', ''), 'a/'));
assert(strcmp(fullfile('/', 'a'), '/a'));
assert(strcmp(fullfile('a', 'b/'), 'a/b/'));
assert(strcmp(fullfile('a//b', 'c'), 'a/b/c'));
assert(strcmp(fullfile('//srv', 'x'), '//srv/x'));
assert(strcmp(fullfile('.', 'a'), './a'));
assert(strcmp(fullfile('a', '.', 'b'), 'a/b'));
assert(isempty(fullfile('', '')) && ischar(fullfile('', '')));
assert(isstring(fullfile("a", 'b')) && fullfile("a", 'b') == "a/b");
c = fullfile('a', {'b', 'c'});
assert(iscell(c) && isequal(c, {'a/b', 'a/c'}));
c = fullfile({'x'; 'y'}, 'z');
assert(isequal(size(c), [2 1]) && strcmp(c{2}, 'y/z'));

% isvector: 2-D with one row or one column
assert(~isvector([]));
assert(~isvector(zeros(1, 1, 3)));
assert(~isvector(zeros(2, 0)));
assert(isvector(zeros(1, 0)));
assert(isvector(zeros(0, 1)));
assert(isvector(5));
assert(isvector('abc'));
assert(isvector({1, 2}));
assert(isvector(struct('a', {1, 2})));
assert(~isvector(ones(2, 3)));

disp('SUCCESS')
