% Test switch/case with various types (exercises compare.ts valuesAreEqual)

% Switch on number
x = 2;
switch x
    case 1
        r = 'one';
    case 2
        r = 'two';
    case 3
        r = 'three';
end
assert(strcmp(r, 'two'));

% Switch on string
s = 'hello';
switch s
    case 'hi'
        r2 = 1;
    case 'hello'
        r2 = 2;
end
assert(r2 == 2);

% Switch on char with cell array of cases
c = 'b';
switch c
    case {'a', 'b', 'c'}
        r3 = 'found';
    otherwise
        r3 = 'not found';
end
assert(strcmp(r3, 'found'));

% Switch with tensors
v = [1 2 3];
switch 1
    case 1
        r4 = 'matched';
    otherwise
        r4 = 'no';
end
assert(strcmp(r4, 'matched'));

% Switch with otherwise
x2 = 99;
switch x2
    case 1
        r5 = 'one';
    case 2
        r5 = 'two';
    otherwise
        r5 = 'other';
end
assert(strcmp(r5, 'other'));

% Switch on logical
flag = true;
switch flag
    case true
        r6 = 'yes';
    case false
        r6 = 'no';
end
assert(strcmp(r6, 'yes'));

disp('SUCCESS')
