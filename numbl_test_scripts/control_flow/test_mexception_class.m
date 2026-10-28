% MException is a class: catch binds an MException object with the
% properties identifier, message, cause, stack and Correction.

% --- class and properties of a caught exception ---
try
    mex_inner(3);
catch ME
end
assert(strcmp(class(ME), 'MException'));
assert(isa(ME, 'MException'));
assert(isobject(ME));
assert(~isstruct(ME));
assert(~isfield(ME, 'message'));
p = properties(ME);
assert(isequal(p(:)', {'identifier', 'message', 'cause', 'stack', 'Correction'}));
assert(strcmp(ME.identifier, 'my:id'));
assert(strcmp(ME.message, 'bad value 3'));
assert(iscell(ME.cause) && isempty(ME.cause));
assert(isempty(ME.Correction));
assert(isstruct(ME.stack));
assert(numel(ME.stack) >= 2 && iscolumn(ME.stack));
assert(strcmp(ME.stack(1).name, 'mex_inner'));
assert(ME.stack(1).line > 0);
assert(~isempty(strfind(ME.stack(1).file, 'test_mexception_class')));

% Properties are read-only.
try
    ME.message = 'changed';
    error('should not get here');
catch E
    assert(strcmp(E.identifier, 'MATLAB:class:SetProhibited'));
end
assert(strcmp(ME.message, 'bad value 3'));

% An exception keeps its class inside containers and arrays.
s.a = ME;
c = {ME};
assert(strcmp(class(s.a), 'MException'));
assert(strcmp(class(c{1}), 'MException'));
arr = [ME, MException('a:b', 'c')];
assert(strcmp(class(arr), 'MException'));
assert(numel(arr) == 2);

% --- top-level error: the stack holds the script frame ---
try
    error('top:id', 'top %s', 'level');
catch ME2
end
assert(strcmp(ME2.identifier, 'top:id'));
assert(strcmp(ME2.message, 'top level'));
assert(numel(ME2.stack) >= 1);
assert(strcmp(ME2.stack(1).name, 'test_mexception_class'));
assert(ME2.stack(1).line == 46);

% --- getReport ---
r = getReport(ME, 'basic');
assert(ischar(r));
assert(strncmp(r, 'Error using test_mexception_class>mex_inner (line', 49));
assert(~isempty(strfind(r, 'bad value 3')));
assert(isempty(strfind(r, 'Error in')));
r = getReport(ME);
assert(~isempty(strfind(r, 'bad value 3')));
assert(~isempty(strfind(r, 'Error in')));
r = getReport(ME, 'extended', 'hyperlinks', 'off');
assert(~isempty(strfind(r, 'Error in')));
r = ME.getReport('basic');
assert(~isempty(strfind(r, 'bad value 3')));
try
    getReport(ME, 'bogus');
    error('should not get here');
catch E
    assert(strcmp(E.identifier, 'MATLAB:MException:getReport:invalidReportOption'));
end

% --- constructor ---
M = MException('a:b', 'x=%d, y=%s', 5, 'z');
assert(strcmp(class(M), 'MException'));
assert(strcmp(M.message, 'x=5, y=z'));
assert(isempty(M.stack) && isequal(size(M.stack), [0 1]));
assert(strcmp(getReport(M), 'x=5, y=z'));
% The message is always a format string, even without values.
M0 = MException('a:b', 'no value %d');
assert(strcmp(M0.message, 'no value '));
Mt = MException('a:b', 'tab\there');
assert(strcmp(Mt.message, sprintf('tab\there')));
Ms = MException("a:b", "string %s", "msg");
assert(ischar(Ms.message) && strcmp(Ms.message, 'string msg'));
try
    MException('a:1', 'x');
    error('should not get here');
catch E
    assert(strcmp(E.identifier, 'MATLAB:MException:badIdentifier'));
end
try
    MException('noColon', 'x');
    error('should not get here');
catch E
    assert(strcmp(E.identifier, 'MATLAB:MException:badIdentifier'));
end
assert(isequal(MException('a:b', 'c'), MException('a:b', 'c')));
assert(~isequal(MException('a:b', 'c'), MException('a:b', 'd')));

% --- addCause ---
M2 = addCause(M, MException('c:one', 'first cause'));
M2 = M2.addCause(MException('c:two', 'second cause'));
assert(numel(M2.cause) == 2);
assert(size(M2.cause, 1) == 2);
assert(strcmp(class(M2.cause{1}), 'MException'));
assert(strcmp(M2.cause{2}.message, 'second cause'));
assert(isempty(M.cause));
r = getReport(M2);
assert(~isempty(strfind(r, 'Caused by:')));
assert(~isempty(strfind(r, 'first cause')));
assert(isempty(strfind(getReport(M2, 'basic'), 'Caused by:')));
try
    addCause(M, 5);
    error('should not get here');
catch E
    assert(strcmp(E.identifier, 'MATLAB:class:RequireClass'));
end

% --- throw keeps the object and records the throw site ---
try
    M2.throw();
catch E3
end
assert(strcmp(class(E3), 'MException'));
assert(strcmp(E3.identifier, 'a:b'));
assert(numel(E3.cause) == 2);
assert(numel(E3.stack) >= 1);
assert(E3.stack(1).line == 124);
try
    throw(struct('message', 'x'));
    error('should not get here');
catch E
    assert(strcmp(E.identifier, 'MATLAB:UndefinedFunction'));
end

% --- rethrow keeps the original stack ---
try
    try
        mex_inner(4);
    catch E5
        rethrow(E5);
    end
catch E6
end
assert(strcmp(E6.message, 'bad value 4'));
assert(strcmp(E6.stack(1).name, 'mex_inner'));
assert(E6.stack(1).line == E5.stack(1).line);

% --- throwAsCaller omits the calling function's frame ---
try
    mex_thrower_ac();
catch E7
end
assert(strcmp(E7.identifier, 'ac:id'));
assert(strcmp(E7.stack(1).name, 'mex_thrower_ac'));

% --- error(struct), error(ME), rethrow(struct) ---
try
    error(struct('message', 'smsg', 'identifier', 's:id'));
catch E9
end
assert(strcmp(class(E9), 'MException'));
assert(strcmp(E9.identifier, 's:id'));
assert(strcmp(E9.message, 'smsg'));
assert(numel(E9.stack) >= 1);
try
    error(M);
catch E8
end
assert(strcmp(E8.identifier, 'MATLAB:error:invalidMessageType'));
try
    rethrow(struct('message', 'rmsg', 'identifier', 'r:id'));
catch E10
end
assert(strcmp(E10.identifier, 'r:id'));
assert(strcmp(E10.message, 'rmsg'));
% Without a stack field, the stack is that of the caller of the function
% that calls rethrow.
E10b = mex_rethrow_struct();
assert(strcmp(E10b.identifier, 'r:id'));
assert(strcmp(E10b.stack(1).name, 'test_mexception_class'));
st = struct('file', 'f.m', 'name', 'fn', 'line', 7);
try
    rethrow(struct('message', 'with stack', 'identifier', 'r:id', 'stack', st));
catch E11
end
assert(numel(E11.stack) == 1);
assert(strcmp(E11.stack(1).name, 'fn'));
assert(E11.stack(1).line == 7);
% Empty error structs and empty messages are no-ops.
error(struct('message', {}));
error('');

% --- errors raised by the interpreter carry identifiers ---
try
    mex_undefined_function_xyz(1);
catch E12
end
assert(strcmp(class(E12), 'MException'));
assert(strcmp(E12.identifier, 'MATLAB:UndefinedFunction'));

disp('SUCCESS');

function mex_inner(n)
    error('my:id', 'bad value %d', n);
end

function mex_thrower_ac()
    mex_ac_helper();
end

function mex_ac_helper()
    ME = MException('ac:id', 'as caller');
    throwAsCaller(ME);
end

function E = mex_rethrow_struct()
    try
        rethrow(struct('message', 'rmsg', 'identifier', 'r:id'));
    catch E
    end
end
