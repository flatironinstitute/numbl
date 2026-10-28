% Property and method access attributes (Access, GetAccess, SetAccess,
% immutable, class lists, Dependent without a setter), Abstract and Sealed.

a = AttrBase(3);

% --- reading restricted properties from outside the class
expectError(@() a.priv, 'MATLAB:class:GetProhibited', ...
    'No public property ''priv'' for class ''''AttrBase''''.');
expectError(@() a.prot, 'MATLAB:class:GetProhibited', ...
    'No public property ''prot'' for class ''''AttrBase''''.');
expectError(@() a.wo, 'MATLAB:class:GetProhibited', ...
    'No public property ''wo'' for class ''''AttrBase''''.');
expectError(@() a.nosuch, 'MATLAB:noSuchMethodOrField', ...
    'Unrecognized method, property, or field ''nosuch'' for class ''AttrBase''.');

% --- setting restricted properties from outside the class
try
    a.priv = 1;
    error('no error');
catch e
    assert(strcmp(e.identifier, 'MATLAB:class:SetProhibited'));
    assert(strcmp(e.message, 'Setting the ''priv'' property of class ''''AttrBase'''' is not supported.'));
end
try
    a.ro = 1;
    error('no error');
catch e
    assert(strcmp(e.identifier, 'MATLAB:class:SetProhibited'));
    assert(strcmp(e.message, 'Unable to set the ''ro'' property of class ''''AttrBase'''' because it is read-only.'));
end
try
    a.imm = 1;
    error('no error');
catch e
    assert(strcmp(e.identifier, 'MATLAB:class:SetProhibited'));
end
try
    a.dep = 1;
    error('no error');
catch e
    assert(strcmp(e.identifier, 'MATLAB:class:noSetMethod'));
    assert(strcmp(e.message, 'In class ''AttrBase'', no set method is defined for dependent property ''dep''. A dependent property needs a set method to assign its value.'));
end
try
    a.nosuch = 1;
    error('no error');
catch e
    assert(strcmp(e.identifier, 'MATLAB:noPublicFieldForClass'));
    assert(strcmp(e.message, 'Unrecognized property ''nosuch'' for class ''AttrBase''.'));
end

% --- public access and access from inside the class
a.pub = 2;
assert(a.pub == 2);
assert(a.dep == 20);
assert(a.ro == 4);
assert(a.imm == 4);          % set twice in the constructor
a.wo = 60;                   % set-only from outside
assert(a.readWo() == 60);
assert(a.readPriv() == 2);
a.setPriv(22);
assert(a.readPriv() == 22);
a.setRo(11);
assert(a.ro == 11);
f = a.privGetter();          % an anonymous function keeps its class access
assert(f() == 22);
assert(a.viaLocal() == 22);  % so does a local function of the classdef file

% immutable: only in the defining class's constructor
expectError(@() a.setImm(1), 'MATLAB:class:SetProhibited', ...
    'Unable to set the ''imm'' property of class ''''AttrBase'''' because it is read-only.');
try
    s = AttrSub(true);
    error('no error');
catch e
    assert(strcmp(e.identifier, 'MATLAB:class:SetProhibited'));
    assert(strcmp(e.message, 'Unable to set the ''imm'' property of class ''''AttrSub'''' because it is read-only.'));
end

% --- methods
assert(a.callPriv() == 84);
expectError(@() a.privMeth(), 'MATLAB:class:MethodRestricted', ...
    'Cannot access method ''privMeth'' in class ''AttrBase''.');
expectError(@() privMeth(a), 'MATLAB:class:MethodRestricted', ...
    'Cannot access method ''privMeth'' in class ''AttrBase''.');
expectError(@() a.protMeth(), 'MATLAB:class:MethodRestricted', ...
    'Cannot access method ''protMeth'' in class ''AttrBase''.');
expectError(@() AttrBase.privStatic(), 'MATLAB:class:MethodRestricted', ...
    'Cannot access method ''privStatic'' in class ''AttrBase''.');
assert(AttrBase.callPrivStatic() == 44);
expectError(@() AttrPrivCtor(1), 'MATLAB:class:MethodRestricted', ...
    'Cannot access method ''AttrPrivCtor'' in class ''AttrPrivCtor''.');
p = AttrPrivCtor.make(5);
assert(p.v == 5);

% --- subclasses: protected members are visible, private ones are not
b = AttrSub();
assert(b.readProt() == 3);
assert(b.readProtOf(AttrBase(1)) == 3);
assert(b.callProt() == 86);
assert(b.callPriv() == 84);  % AttrBase's own method on a subclass object
b.setSprot(70);
assert(b.sprot == 70);
expectError(@() b.readPriv2(), 'MATLAB:noSuchMethodOrField', ...
    'Unrecognized method, property, or field ''priv'' for class ''AttrSub''.');
expectError(@() b.callPriv2(), 'MATLAB:noSuchMethodOrField', ...
    'Unrecognized method, property, or field ''privMeth'' for class ''AttrSub''.');
expectError(@() b.setRo2(1), 'MATLAB:class:SetProhibited', ...
    'Unable to set the ''ro'' property of class ''''AttrSub'''' because it is read-only.');
try
    b.sprot = 1;
    error('no error');
catch e
    assert(strcmp(e.identifier, 'MATLAB:class:SetProhibited'));
end

% --- class lists: Access = {?AttrFriend}
assert(a.readFriendly() == 8);          % the defining class keeps access
assert(AttrFriend.peek(a) == 8);
AttrFriend.poke(a, 80);
assert(AttrFriend.peek(a) == 80);
expectError(@() a.friendly, 'MATLAB:class:GetProhibited', ...
    'No public property ''friendly'' for class ''''AttrBase''''.');
expectError(@() AttrFriend.peekPriv(a), 'MATLAB:class:GetProhibited', ...
    'No public property ''priv'' for class ''''AttrBase''''.');

% --- Abstract and Sealed
expectError(@() AttrAbstract(), 'MATLAB:class:abstract', ...
    'Abstract classes cannot be instantiated. Class ''AttrAbstract'' defines abstract methods and/or properties.');
expectError(@() AttrPartial(), 'MATLAB:class:abstract', ...
    ['Abstract classes cannot be instantiated. Class ''AttrPartial'' inherits abstract methods or properties but does not implement them. ' ...
     'Use matlab.metadata.abstractDetails to see the list of methods and properties that ''AttrPartial'' must implement if you do not intend the class to be abstract.']);
c = AttrConcrete();
assert(c.doIt() == 6);
assert(c.sealedM() == 1);
expectError(@() AttrSealedSub(), 'MATLAB:class:sealed', ...
    'Class ''AttrSealed'' is Sealed and may not be used as a superclass.');
expectError(@() AttrOverrideSealed(), 'MATLAB:class:methodOverrideSealed', ...
    'Method ''sealedM'' in class ''AttrOverrideSealed'' conflicts with the sealed method in superclass ''AttrAbstract''. Overriding a sealed method is not supported.');

disp('SUCCESS');

function expectError(f, id, msg)
try
    f();
catch e
    assert(strcmp(e.identifier, id), sprintf('identifier: %s', e.identifier));
    assert(strcmp(e.message, msg), sprintf('message: %s', e.message));
    return;
end
error('expected an error: %s', id);
end
