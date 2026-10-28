% class, isa, the type predicates and size queries across every kind of
% value (expected values checked against MATLAB).

names = {};
vals = {};
names{end+1} = 'dscalar'; vals{end+1} = 1;
names{end+1} = 'drow'; vals{end+1} = [1 2 3];
names{end+1} = 'dempty'; vals{end+1} = [];
names{end+1} = 'd3d'; vals{end+1} = zeros(2,3,4);
names{end+1} = 'lscalar'; vals{end+1} = true;
names{end+1} = 'lrow'; vals{end+1} = [true false];
names{end+1} = 'cscalar'; vals{end+1} = 1+2i;
names{end+1} = 'crow'; vals{end+1} = [1+2i 3];
names{end+1} = 'chrow'; vals{end+1} = 'abc';
names{end+1} = 'chempty'; vals{end+1} = '';
names{end+1} = 'chmat'; vals{end+1} = ['ab';'cd'];
names{end+1} = 'str'; vals{end+1} = "abc";
names{end+1} = 'strarr'; vals{end+1} = ["a" "b"];
names{end+1} = 'strempty'; vals{end+1} = string([]);
names{end+1} = 'cellrow'; vals{end+1} = {1,2};
names{end+1} = 'cellempty'; vals{end+1} = {};
names{end+1} = 'struct1'; vals{end+1} = struct('a',1);
names{end+1} = 'structarr'; vals{end+1} = struct('a',{1,2});
names{end+1} = 'structempty'; vals{end+1} = struct('a',{});
names{end+1} = 'fh'; vals{end+1} = @sin;
names{end+1} = 'anon'; vals{end+1} = @(x) x+1;
names{end+1} = 'sp'; vals{end+1} = sparse(eye(2));
names{end+1} = 'spc'; vals{end+1} = sparse([1+2i 0]);
names{end+1} = 'dict'; vals{end+1} = dictionary('a',1);
names{end+1} = 'dictempty'; vals{end+1} = dictionary();
names{end+1} = 'objval'; vals{end+1} = KindVal_();
names{end+1} = 'objsub'; vals{end+1} = KindSub_();
names{end+1} = 'objhandle'; vals{end+1} = KindHandle_();
names{end+1} = 'objarr'; vals{end+1} = [KindVal_() KindVal_()];
names{end+1} = 'objdouble'; vals{end+1} = KindDouble_(3);
names{end+1} = 'enum'; vals{end+1} = KindEnum_.red;
names{end+1} = 'intenum'; vals{end+1} = KindIntEnum_.hi;
names{end+1} = 'mexc'; vals{end+1} = MException('a:b','msg');

% name, class, [isnumeric isfloat isinteger islogical isobject isreal isempty], numel, size
expected = {
    'dscalar', 'double', [1 1 0 0 0 1 0], 1, [1 1]
    'drow', 'double', [1 1 0 0 0 1 0], 3, [1 3]
    'dempty', 'double', [1 1 0 0 0 1 1], 0, [0 0]
    'd3d', 'double', [1 1 0 0 0 1 0], 24, [2 3 4]
    'lscalar', 'logical', [0 0 0 1 0 1 0], 1, [1 1]
    'lrow', 'logical', [0 0 0 1 0 1 0], 2, [1 2]
    'cscalar', 'double', [1 1 0 0 0 0 0], 1, [1 1]
    'crow', 'double', [1 1 0 0 0 0 0], 2, [1 2]
    'chrow', 'char', [0 0 0 0 0 1 0], 3, [1 3]
    'chempty', 'char', [0 0 0 0 0 1 1], 0, [0 0]
    'chmat', 'char', [0 0 0 0 0 1 0], 4, [2 2]
    'str', 'string', [0 0 0 0 1 0 0], 1, [1 1]
    'strarr', 'string', [0 0 0 0 1 0 0], 2, [1 2]
    'strempty', 'string', [0 0 0 0 1 0 1], 0, [0 0]
    'cellrow', 'cell', [0 0 0 0 0 0 0], 2, [1 2]
    'cellempty', 'cell', [0 0 0 0 0 0 1], 0, [0 0]
    'struct1', 'struct', [0 0 0 0 0 0 0], 1, [1 1]
    'structarr', 'struct', [0 0 0 0 0 0 0], 2, [1 2]
    'structempty', 'struct', [0 0 0 0 0 0 1], 0, [0 0]
    'fh', 'function_handle', [0 0 0 0 0 0 0], 1, [1 1]
    'anon', 'function_handle', [0 0 0 0 0 0 0], 1, [1 1]
    'sp', 'double', [1 1 0 0 0 1 0], 4, [2 2]
    'spc', 'double', [1 1 0 0 0 0 0], 2, [1 2]
    'dict', 'dictionary', [0 0 0 0 1 0 0], 1, [1 1]
    'dictempty', 'dictionary', [0 0 0 0 1 0 0], 1, [1 1]
    'objval', 'KindVal_', [0 0 0 0 1 0 0], 1, [1 1]
    'objsub', 'KindSub_', [0 0 0 0 1 0 0], 1, [1 1]
    'objhandle', 'KindHandle_', [0 0 0 0 1 0 0], 1, [1 1]
    'objarr', 'KindVal_', [0 0 0 0 1 0 0], 2, [1 2]
    'objdouble', 'KindDouble_', [1 1 0 0 1 1 0], 1, [1 1]
    'enum', 'KindEnum_', [0 0 0 0 1 0 0], 1, [1 1]
    'intenum', 'KindIntEnum_', [1 0 1 0 1 1 0], 1, [1 1]
    'mexc', 'MException', [0 0 0 0 1 0 0], 1, [1 1]
};

preds = {@isnumeric, @isfloat, @isinteger, @islogical, @isobject, @isreal, @isempty};
for k = 1:numel(names)
    v = vals{k};
    row = expected(k, :);
    assert(strcmp(row{1}, names{k}));
    cls = row{2};
    flags = row{3};
    assert(strcmp(class(v), cls), [names{k} ': class']);
    for p = 1:numel(preds)
        assert(preds{p}(v) == flags(p), [names{k} ': ' func2str(preds{p})]);
    end
    % isnumeric/isfloat/isinteger agree with isa's categories
    assert(isa(v, 'numeric') == flags(1), [names{k} ': isa numeric']);
    assert(isa(v, 'float') == flags(2), [names{k} ': isa float']);
    assert(isa(v, 'integer') == flags(3), [names{k} ': isa integer']);
    assert(isa(v, cls), [names{k} ': isa own class']);
    assert(numel(v) == row{4}, [names{k} ': numel']);
    assert(isequal(size(v), row{5}), [names{k} ': size']);
    assert(ndims(v) == numel(row{5}), [names{k} ': ndims']);
    % cellfun('isclass') compares the class name exactly
    assert(cellfun('isclass', {v}, cls), [names{k} ': isclass']);
    assert(~cellfun('isclass', {v}, 'numeric'), [names{k} ': isclass numeric']);
    validateattributes(v, {cls}, {});
end

% isa with built-in class names
s = struct('a', {1, 2});
assert(isa(s, 'struct'));
assert(isa(struct('a', {}), 'struct'));
assert(isa(sparse(eye(2)), 'double'));
assert(isa(sparse(eye(2)), 'numeric'));
assert(isa(sparse(eye(2)), 'float'));
assert(isa(1, 'float') && isa([], 'float') && isa(1i, 'float'));
assert(~isa(true, 'numeric') && ~isa('a', 'numeric') && ~isa(1, 'integer'));
assert(isa(dictionary('a', 1), 'dictionary'));
assert(isa(@sin, 'function_handle'));
assert(isa("a", 'string') && isa(["a" "b"], 'string'));
assert(~isa(s, 'cell') && ~isa({}, 'struct'));

% isa walks superclasses, including built-in ones
assert(isa(KindSub_(), 'KindVal_'));
assert(~isa(KindVal_(), 'KindSub_'));
assert(isa(KindHandle_(), 'handle'));
assert(~isa(KindVal_(), 'handle'));
d = KindDouble_(3);
assert(isa(d, 'double') && isa(d, 'numeric') && isa(d, 'float'));
assert(~isa(d, 'integer'));
e = KindIntEnum_.hi;
assert(isa(e, 'int32') && isa(e, 'numeric') && isa(e, 'integer'));
assert(~isa(e, 'float') && ~isa(e, 'double'));
assert(~isa(KindEnum_.red, 'numeric'));

% cellfun('isclass'): exact class names, no categories or superclasses
c = {1, true, 'a', s, KindSub_(), {}, sparse(1), "s", dictionary('a', 1)};
assert(isequal(cellfun('isclass', c, 'double'), [1 0 0 0 0 0 1 0 0]));
assert(isequal(cellfun('isclass', c, 'struct'), [0 0 0 1 0 0 0 0 0]));
assert(isequal(cellfun('isclass', c, 'KindSub_'), [0 0 0 0 1 0 0 0 0]));
assert(isequal(cellfun('isclass', c, 'KindVal_'), [0 0 0 0 0 0 0 0 0]));
assert(isequal(cellfun('isclass', c, 'dictionary'), [0 0 0 0 0 0 0 0 1]));

% whos reports the class of struct arrays and dictionaries
dct = dictionary('a', 1);
w = whos('s');
assert(strcmp(w.class, 'struct'));
w = whos('dct');
assert(strcmp(w.class, 'dictionary'));
% and, without an output, does not set ans
clear ans
whos('s')
who('s')
assert(~exist('ans', 'var'));

% validateattributes: class names match with isa (superclasses count);
% 'numeric' is the only category it accepts
validateattributes(KindSub_(), {'KindVal_'}, {});
validateattributes(KindDouble_(1), {'numeric'}, {});
validateattributes(KindIntEnum_.lo, {'numeric'}, {});
validateattributes(sparse(eye(2)), {'numeric'}, {});
validateattributes(s, {'struct'}, {});
ok = true;
try
    validateattributes(1, {'float'}, {});
catch
    ok = false;
end
assert(~ok);
ok = true;
try
    validateattributes(KindVal_(), {'numeric'}, {});
catch
    ok = false;
end
assert(~ok);

% the same in a loop (compiled by the JIT)
n = 0;
x = [1 2];
for k = 1:3
    n = n + isa(k, 'float') + isa(x, 'numeric') + isa(k, 'integer') + isfloat(x) + isinteger(k);
end
assert(n == 9);

% length of an empty array is 0 whatever its shape
assert(length(zeros(3, 0)) == 0);
assert(length(char(zeros(3, 0))) == 0);
assert(length(cell(0, 4)) == 0);
assert(length(struct('a', cell(2, 3))) == 3);

disp('SUCCESS')
