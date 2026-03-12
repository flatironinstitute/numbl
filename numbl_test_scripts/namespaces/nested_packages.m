% Test nested packages

% Call function from utils package
x = utils.double_it(7);
assert(x == 14);

% Call function from nested utils.string package
s = utils.string.reverse('hello');
assert(strcmp(s, 'olleh'));

s2 = utils.string.reverse('MATLAB');
assert(strcmp(s2, 'BALTAM'));

disp('SUCCESS')
