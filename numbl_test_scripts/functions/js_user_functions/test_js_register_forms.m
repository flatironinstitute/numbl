% register(function ...) and register({apply, help}) forms
[m, r] = mystats([1 2 3 10]);
assert(m == 4 && r == 9);
assert(mystats([2 4]) == 3);

h = help('mystats');
assert(contains(h, 'Mean and range of a vector.'));

% A function that returns nothing can be called as a statement but errors
% when an output is requested.
mynoout(1);
try
    y = mynoout(1);
    assert(false, 'expected an error');
catch e
    assert(contains(e.message, 'Too many output arguments'));
end

disp('SUCCESS')
