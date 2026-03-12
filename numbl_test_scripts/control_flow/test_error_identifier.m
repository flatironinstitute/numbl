% Test error() with message identifier
% error('id:sub', 'msg', ...) should set identifier and format message

% error with identifier and sprintf-style formatting
try
    error('myid:myerr', 'value is %d', 42);
catch e
    assert(strcmp(e.message, 'value is 42'), 'error message should be formatted');
    assert(strcmp(e.identifier, 'myid:myerr'), 'error identifier should be set');
end

% error with identifier, no extra args
try
    error('test:err', 'simple message');
catch e
    assert(strcmp(e.message, 'simple message'), 'error message without format args');
    assert(strcmp(e.identifier, 'test:err'), 'error identifier without format args');
end

% error with just a message (no identifier) should still work
try
    error('plain error');
catch e
    assert(strcmp(e.message, 'plain error'), 'plain error message');
end

disp('SUCCESS');
