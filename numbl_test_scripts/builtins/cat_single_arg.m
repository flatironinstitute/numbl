% Test that cat with only a dim argument returns empty
result = cat(1);
assert(isempty(result));
assert(isequal(size(result), [0 0]));

result2 = cat(2);
assert(isempty(result2));

fprintf('SUCCESS\n');
