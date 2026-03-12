% Test name-value arguments in arguments blocks

% Test 1: All defaults
r1 = myFunc(10);
assert(r1 == 10 + 1 + 100);

% Test 2: Override one name-value pair
r2 = myFunc(10, 'alpha', 5);
assert(r2 == 10 + 5 + 100);

% Test 3: Override both
r3 = myFunc(10, 'alpha', 5, 'beta', 200);
assert(r3 == 10 + 5 + 200);

% Test 4: Override in different order
r4 = myFunc(10, 'beta', 200, 'alpha', 5);
assert(r4 == 10 + 5 + 200);

fprintf('SUCCESS\n');

function result = myFunc(x, opts)
    arguments
        x
        opts.alpha = 1
        opts.beta = 100
    end
    result = x + opts.alpha + opts.beta;
end
