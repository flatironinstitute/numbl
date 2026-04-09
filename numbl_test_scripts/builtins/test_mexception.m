% Test MException constructor and related functions

% Basic two-argument form
ME = MException('myid:bad', 'something went wrong');
assert(strcmp(ME.identifier, 'myid:bad'), 'identifier should match');
assert(strcmp(ME.message, 'something went wrong'), 'message should match');

% Formatted message form
ME2 = MException('myid:fmt', 'value is %d', 42);
assert(strcmp(ME2.identifier, 'myid:fmt'), 'fmt identifier');
assert(strcmp(ME2.message, 'value is 42'), 'formatted message');

% Multiple format args
ME3 = MException('myid:multi', '%s = %d', 'x', 7);
assert(strcmp(ME3.message, 'x = 7'), 'multi-arg formatted message');

% Float formatting
ME4 = MException('myid:f', 'pi=%.2f', 3.14159);
assert(strcmp(ME4.message, 'pi=3.14'), 'float-formatted message');

% Empty identifier is allowed
ME5 = MException('', 'no id here');
assert(strcmp(ME5.identifier, ''), 'empty identifier');
assert(strcmp(ME5.message, 'no id here'), 'message with empty id');

% throw(ME) — throws and is catchable
caught = false;
try
    throw(ME);
catch err
    caught = true;
    assert(strcmp(err.identifier, 'myid:bad'), 'thrown identifier');
    assert(strcmp(err.message, 'something went wrong'), 'thrown message');
end
assert(caught, 'throw should have triggered catch');

% throwAsCaller(ME) — also throws and is catchable
caught2 = false;
try
    throwAsCaller(ME2);
catch err2
    caught2 = true;
    assert(strcmp(err2.identifier, 'myid:fmt'), 'thrownAsCaller identifier');
    assert(strcmp(err2.message, 'value is 42'), 'thrownAsCaller message');
end
assert(caught2, 'throwAsCaller should have triggered catch');

% getReport — basic form returns just the message
report_basic = getReport(ME, 'basic');
assert(ischar(report_basic), 'getReport returns char');
assert(strcmp(report_basic, 'something went wrong'), 'basic report is message');

% getReport — single-arg form (defaults to extended)
report_default = getReport(ME);
assert(ischar(report_default), 'getReport default returns char');
% For an MException constructed with no stack, basic and extended both contain the message
assert(contains(report_default, 'something went wrong'), 'default report contains message');

disp('SUCCESS');
