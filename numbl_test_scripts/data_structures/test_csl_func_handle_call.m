% Test CSL expansion in function handle calls: f(c{:})
% When a cell array is expanded with {:} and passed to a function handle,
% the elements should be spread as individual arguments.

function test_csl_func_handle_call()

% Basic case: function handle with cell expansion
f = @(a, b) a + b;
c = {3, 4};
result = f(c{:});
assert(result == 7, 'f(c{:}) should expand cell and pass as individual args');

% Function handle with more arguments
g = @(a, b, c) a * b + c;
args = {2, 3, 5};
result = g(args{:});
assert(result == 11, 'g(args{:}) should expand 3-element cell');

% Function handle with mixed args and CSL
h = @(a, b, c) a + b + c;
partial = {10, 20};
result = h(partial{:}, 30);
assert(result == 60, 'h(partial{:}, 30) should expand partial cell and append arg');

% Function handle stored in a variable from another function
adder = @my_add;
pair = {100, 200};
result = adder(pair{:});
assert(result == 300, 'adder(pair{:}) should work with named function handle');

fprintf('SUCCESS\n');

end

function result = my_add(a, b)
    result = a + b;
end
