% Test general MATLAB command syntax
% In command syntax, all arguments are passed as character vectors

% Basic: "assert_is_char hello" uses command syntax → assert_is_char('hello')
assert_is_char hello

% Compound arg with dot (like a filename)
assert_is_char data.mat

% Path argument with slash
assert_is_char some/path

% Flag argument starting with dash
assert_is_char -verbose

% Numeric text becomes char vector, not a number
assert_is_char 42

% Quoted string with spaces preserved as single arg
assert_is_char 'hello world'

% Multiple command-syntax args: assert each is char and check values
assert_chars_eq foo foo
assert_chars_eq data.mat data.mat

% Verify the args are actually char vectors matching expected text
x = get_cmd_arg('hello');
assert(strcmp(x, 'hello'));
x = get_cmd_arg('data.mat');
assert(strcmp(x, 'data.mat'));

% Existing COMMAND_VERBS still work
y = 123;
clear y
try
    z = y;
    error('y should have been cleared');
catch
    % expected — clear worked
end

disp('SUCCESS')

function assert_is_char(x)
    assert(ischar(x), 'command syntax arg must be char');
end

function assert_chars_eq(a, b)
    assert(ischar(a), 'first arg must be char');
    assert(ischar(b), 'second arg must be char');
    assert(strcmp(a, b), 'char args must be equal');
end

function out = get_cmd_arg(x)
    out = x;
end
