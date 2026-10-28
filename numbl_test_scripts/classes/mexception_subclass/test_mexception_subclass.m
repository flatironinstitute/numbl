% A user subclass of MException: constructed through obj@MException,
% thrown and caught as its own class, with the MException methods.

e = MyErr_('my:err', 'custom failure', 42);
assert(strcmp(class(e), 'MyErr_'));
assert(isa(e, 'MyErr_'));
assert(isa(e, 'MException'));
assert(strcmp(e.identifier, 'my:err'));
assert(strcmp(e.message, 'custom failure'));
assert(e.Extra == 42);
assert(isempty(e.stack));
assert(iscell(e.cause) && isempty(e.cause));
p = properties(e);
assert(any(strcmp(p, 'Extra')));
assert(any(strcmp(p, 'identifier')));
assert(any(strcmp(p, 'stack')));
assert(strcmp(e.describe(), 'custom failure [42]'));
assert(strcmp(getReport(e), 'custom failure'));

% throw keeps the subclass and its properties.
try
    throw(e);
catch ME
end
assert(strcmp(class(ME), 'MyErr_'));
assert(isa(ME, 'MException'));
assert(strcmp(ME.identifier, 'my:err'));
assert(ME.Extra == 42);
assert(numel(ME.stack) >= 1);
assert(strcmp(describe(ME), 'custom failure [42]'));

% Inherited methods called with dot syntax.
try
    e.throw();
catch ME2
end
assert(strcmp(class(ME2), 'MyErr_'));
r = ME2.getReport('basic');
assert(~isempty(strfind(r, 'custom failure')));
e2 = e.addCause(MException('c:x', 'the cause'));
assert(strcmp(class(e2), 'MyErr_'));
assert(numel(e2.cause) == 1);

% rethrow keeps the subclass too.
try
    try
        throw(e);
    catch ME3
        rethrow(ME3);
    end
catch ME4
end
assert(strcmp(class(ME4), 'MyErr_'));
assert(ME4.Extra == 42);

% methods lists the subclass's own methods and the inherited MException
% methods, but not the superclass constructor
m = methods('MyErr_');
assert(any(strcmp(m, 'MyErr_')));
assert(any(strcmp(m, 'describe')));
assert(any(strcmp(m, 'getReport')));
assert(any(strcmp(m, 'throw')));
assert(~any(strcmp(m, 'MException')));
assert(ismethod(ME4, 'rethrow'));

disp('SUCCESS');
