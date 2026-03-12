% Test feval and builtin functions (specialBuiltins.ts coverage)

% feval with function handle
f = @sin;
assert(abs(feval(f, pi/2) - 1) < 1e-10);

% feval with string name
assert(abs(feval('cos', 0) - 1) < 1e-10);

% feval with char name
assert(abs(feval('sqrt', 4) - 2) < 1e-10);

% builtin with string name
x = builtin('zeros', 2, 3);
assert(isequal(size(x), [2 3]));
assert(all(all(x == 0)));

% builtin with char name
y = builtin('ones', 1, 4);
assert(isequal(size(y), [1 4]));
assert(all(y == 1));

% fileparts
[d, n, e] = fileparts('/home/user/test.m');
assert(strcmp(d, '/home/user'));
assert(strcmp(n, 'test'));
assert(strcmp(e, '.m'));

% fileparts with no extension
[d2, n2, e2] = fileparts('/home/user/readme');
assert(strcmp(d2, '/home/user'));
assert(strcmp(n2, 'readme'));
assert(strcmp(e2, ''));

% fileparts with no directory
[d3, n3, e3] = fileparts('test.m');
assert(strcmp(d3, ''));
assert(strcmp(n3, 'test'));
assert(strcmp(e3, '.m'));

% fullfile
p = fullfile('home', 'user', 'test.m');
assert(strcmp(p, 'home/user/test.m'));

disp('SUCCESS');
