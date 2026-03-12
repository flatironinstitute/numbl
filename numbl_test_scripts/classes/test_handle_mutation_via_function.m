% Test that handle objects are mutated when passed to functions
c = Counter_();
assert(c.Value == 0);

increment_counter(c, 5);
assert(c.Value == 5, 'handle object should be mutated by function');

increment_counter(c, 3);
assert(c.Value == 8, 'handle object should accumulate mutations');

reset_via_function(c);
assert(c.Value == 0, 'handle object should be reset by function');

fprintf('SUCCESS\n');

function increment_counter(obj, amount)
    obj.increment(amount);
end

function reset_via_function(obj)
    obj.reset();
end
