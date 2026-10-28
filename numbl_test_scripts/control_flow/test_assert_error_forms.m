% assert / error message forms: which arguments are formatted, and the
% identifier each form sets. Also exercised inside a loop, where the JIT
% raises them.

% assert(cond, fmt, args...) formats the message
try
    assert(false, 'value %d of %d', 3, 4);
catch e
    assert(strcmp(e.message, 'value 3 of 4'));
    assert(isempty(e.identifier));
end

% assert(cond, id, fmt, args...) sets the identifier
try
    assert(false, 'pkg:check', 'bad %s', 'input');
catch e
    assert(strcmp(e.message, 'bad input'));
    assert(strcmp(e.identifier, 'pkg:check'));
end

% assert(cond, id, msg) with no further arguments
try
    assert(false, 'pkg:check', 'plain');
catch e
    assert(strcmp(e.message, 'plain'));
    assert(strcmp(e.identifier, 'pkg:check'));
end

% assert(cond, msg) takes the message literally
try
    assert(false, 'literal %d');
catch e
    assert(strcmp(e.message, 'literal %d'));
end

% error(id, msg) formats the message even with no arguments
try
    error('pkg:id', 'rate 50%%');
catch e
    assert(strcmp(e.message, 'rate 50%'));
    assert(strcmp(e.identifier, 'pkg:id'));
end

% error(msg) takes the message literally
try
    error('rate 50%%');
catch e
    assert(strcmp(e.message, 'rate 50%%'));
    assert(isempty(e.identifier));
end

% The same forms raised from a loop
for k = 1:5
    try
        assert(k < 3, 'pkg:loop', 'k=%d', k);
    catch e
        assert(strcmp(e.message, sprintf('k=%d', k)));
        assert(strcmp(e.identifier, 'pkg:loop'));
        break;
    end
end
assert(k == 3);

disp('SUCCESS');
