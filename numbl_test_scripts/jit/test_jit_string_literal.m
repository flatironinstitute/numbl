% Test that JIT correctly handles double-quoted string literals in functions

function y = getString(x)
    y = "hello";
end

function y = getConcat(a, b)
    y = "prefix_" + a;
end

function y = getBranch(x)
    if x > 0
        y = "positive";
    else
        y = "negative";
    end
end

function y = getEscaped(x)
    y = "say ""hi""";
end

% Basic string return
assert(strcmp(getString(1), 'hello'));

% String in branch
assert(strcmp(getBranch(1), 'positive'));
assert(strcmp(getBranch(-1), 'negative'));

% Escaped quotes
assert(strcmp(getEscaped(1), 'say "hi"'));

% Verify class is string (not char)
assert(strcmp(class(getString(1)), 'string'));

disp('SUCCESS');
