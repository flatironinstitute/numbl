% Constant properties: evaluated once per class, read as ClassName.Prop,
% obj.Prop and inside methods, and never assignable.

assert(ConstHolder.K == 7);
assert(ConstHolder.K2 == 14);
assert(ConstHolder.K == 7);
% The default was evaluated once, whatever the number of reads.
assert(ConstHolder.counter(0) == 1);

% Indexing a constant
assert(ConstHolder.V(2) == 20);
assert(isequal(ConstHolder.V(end), 30));
assert(strcmp(ConstHolder.C{2}, 'two'));
assert(ConstHolder.S.a == 4);

% Through an instance, inside methods, and inherited
obj = ConstHolder(5);
assert(obj.K == 7);
assert(obj.V(3) == 30);
assert(obj.sumK() == 7 + 14 + 300);
assert(ConstHolder.getPK() == 300);
ch = ConstChild_();
assert(ch.K2 == 14);
assert(ConstChild_.K == 7);
assert(ConstHolder.counter(0) == 1);

% A handle-object constant is one shared object
h = ConstHolder.H;
h.n = 3;
assert(ConstHolder.H.n == 3);

% A copied constant is independent of the class's value
v = ConstHolder.V;
v(1) = 99;
assert(ConstHolder.V(1) == 10);

% Private constant from outside
try
    x = ConstHolder.PK;
    error('no error');
catch e
    assert(strcmp(e.identifier, 'MATLAB:class:GetProhibited'));
    assert(strcmp(e.message, 'No public property ''PK'' for class ''''ConstHolder''''.'));
end

% Constants cannot be assigned
try
    obj.K = 1;
    error('no error');
catch e
    assert(strcmp(e.identifier, 'MATLAB:class:SetProhibited'));
    assert(strcmp(e.message, 'Unable to set the ''K'' property of class ''''ConstHolder'''' because it is read-only.'));
end

% Enumeration classes with a Constant property still work
assert(ConstColor_.Default == ConstColor_.Green);
assert(ConstColor_.Red ~= ConstColor_.Green);
c = ConstColor_.Red;
assert(c.Default == ConstColor_.Green);

disp('SUCCESS');
