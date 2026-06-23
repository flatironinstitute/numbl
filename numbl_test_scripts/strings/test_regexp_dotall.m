% MATLAB's regexp/regexprep treat '.' as matching newlines (dotall) by
% default, unlike many engines.
assert(~isempty(regexp(sprintf('{\na\n}'), '\{.*\}', 'once')), ...
    'dot matches newline in regexp');
m = regexp(sprintf('x1\ny2'), 'x.*y', 'match', 'once');
assert(strcmp(m, sprintf('x1\ny')), 'match spans newline');
assert(strcmp(regexprep(sprintf('a\nb'), 'a.b', 'X'), 'X'), ...
    'dot matches newline in regexprep');
disp('SUCCESS');
