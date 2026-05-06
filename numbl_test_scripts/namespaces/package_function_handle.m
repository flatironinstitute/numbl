% Test function handles to package (namespace) functions

% Direct package function call (sanity check)
assert(utils.double_it(3) == 6);

% Function handle to a one-level package function
f1 = @utils.double_it;
assert(f1(4) == 8);

% Function handle to a nested package function
f2 = @utils.string.reverse;
assert(strcmp(f2('hello'), 'olleh'));

% Use a nested-package function handle with cellfun
out = cellfun(@utils.string.reverse, {'abc', 'xyz'}, 'UniformOutput', false);
assert(strcmp(out{1}, 'cba'));
assert(strcmp(out{2}, 'zyx'));

disp('SUCCESS')
