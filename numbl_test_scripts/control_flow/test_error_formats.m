% Test error formatting and warning/assert edge cases

% error with identifier and sprintf formatting
try
    error('mycomp:myerr', 'Value is %d', 42);
catch e
    assert(strcmp(e.message, 'Value is 42'));
    assert(strcmp(e.identifier, 'mycomp:myerr'));
end

% error with identifier, no format args
try
    error('mycomp:myerr', 'simple message');
catch e
    assert(strcmp(e.message, 'simple message'));
end

% error with plain message
try
    error('plain error');
catch e
    assert(strcmp(e.message, 'plain error'));
end

% error with sprintf formatting but no identifier
try
    error('value is %d', 99);
catch e
    assert(strcmp(e.message, 'value is 99'));
end

% warning should not error (it's a no-op)
warning('test warning');
warning('myid:mywarn', 'test %d', 1);

% assert with true
assert(true);
assert(1 == 1);

% assert with custom message
try
    assert(false, 'custom fail message');
catch e
    assert(strcmp(e.message, 'custom fail message'));
end

% assert with default message
try
    assert(false);
catch e
    assert(strcmp(e.message, 'Assertion failed'));
end

disp('SUCCESS');
