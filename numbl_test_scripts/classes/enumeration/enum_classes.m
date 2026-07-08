% Enumeration classes: member access, conversion, comparison, arrays, switch.

%% Member access + class()
t = patchtype.tri;
q = patchtype.quad;
assert(strcmp(class(t), 'patchtype'));

%% Superclass conversion
assert(uint32(t) == 1);
assert(double(q) == 0);
assert(isequal(uint32([t q t]), uint32([1 0 1])));

%% enum vs enum
assert(t == patchtype.tri);
assert(t ~= q);
assert(isequal(t, patchtype.tri));

%% enum vs numeric
assert(t == 1);
assert(t == uint32(1));
assert(q ~= 1);

%% enum vs char/string (matched by member name, scalar result)
assert(t == 'tri');
assert(~(t == 'quad'));
assert(t == "tri");
assert('tri' == t);
assert(~(t == 'xyz'));

%% Array behavior: repmat, indexed assign preserves column, find, all
arr = repmat(t, 3, 1);
arr(2) = q;
assert(isequal(size(arr), [3 1]));
assert(strcmp(class(arr), 'patchtype'));
assert(isequal(arr == 'tri', logical([1;0;1])));
assert(isequal(find(arr == patchtype.tri), [1;3]));
assert(~all(arr == arr(1)));
assert(all(repmat(t, 4, 1) == t));

%% Indexing an enum array yields a scalar enum
e1 = arr(1);
assert(strcmp(class(e1), 'patchtype'));
assert(uint32(e1) == 1);

%% switch on an enum scalar with char case labels
label = pick(arr(1));
assert(strcmp(label, 'is_tri'));
assert(strcmp(pick(q), 'is_quad'));

%% Constructor-as-converter
assert(patchtype(1) == patchtype.tri);
assert(patchtype(0) == patchtype.quad);
assert(patchtype('tri') == patchtype.tri);
assert(patchtype(t) == t);
conv = patchtype([1 0]);
assert(isequal(uint32(conv), uint32([1 0])));

%% Logical superclass
assert(logical(Bool_.Yes));
assert(~logical(Bool_.No));
assert(Bool_.Yes == 1);

%% Plain enumeration (no superclass): compared by member name
d = WeekDays.Tuesday;
assert(strcmp(class(d), 'WeekDays'));
assert(d == WeekDays.Tuesday);
assert(d ~= WeekDays.Monday);
assert(d == 'Tuesday');
assert(strcmp(daykind(d), 'midweek'));

disp('SUCCESS')

function s = pick(p)
switch p
    case 'tri'
        s = 'is_tri';
    case 'quad'
        s = 'is_quad';
    otherwise
        s = 'other';
end
end

function s = daykind(d)
switch d
    case 'Monday'
        s = 'start';
    case {'Tuesday', 'Wednesday', 'Thursday'}
        s = 'midweek';
    otherwise
        s = 'end';
end
end
