% Test ismember with string/char and cell array of strings
% This is a common pattern in chebfun: ismember(fields(s), 'fieldName')

s = struct('a', 1, 'b', 2, 'c', 3);

% Cell array of strings vs single string
result = ismember(fieldnames(s), 'b');
assert(isequal(result, [0; 1; 0]) || isequal(result, logical([0; 1; 0])), ...
    'ismember should find "b" in fieldnames');

% Single string vs cell array of strings
result2 = ismember('b', fieldnames(s));
assert(result2 == true, 'ismember should find "b" in fieldnames');

result3 = ismember('z', fieldnames(s));
assert(result3 == false, 'ismember should not find "z" in fieldnames');

% any(ismember(fields(s), 'a')) pattern from chebfun
assert(any(ismember(fields(s), 'a')), 'should find "a" in fields');
assert(~any(ismember(fields(s), 'z')), 'should not find "z" in fields');

disp('SUCCESS');
