% ?ClassName / metaclass(obj), and properties / fieldnames / isprop /
% ismethod / methods respecting Access and Hidden.

mc = ?MetaBase_;
assert(isa(mc, 'meta.class'));
assert(strcmp(mc.Name, 'MetaBase_'));
assert(~mc.Abstract && ~mc.Sealed && ~mc.Hidden && ~mc.Enumeration);
assert(mc.HandleCompatible);

% PropertyList lists every property of the class itself
names = {mc.PropertyList.Name};
assert(isequal(sort(names), sort({'pub', 'priv', 'prot', 'imm', 'hid', 'K', 'dep', 'tr'})));
assert(isequal(size(mc.PropertyList), [8 1]));
props = mc.PropertyList;
p = props(strcmp(names, 'priv'));
assert(strcmp(p.GetAccess, 'private') && strcmp(p.SetAccess, 'private'));
assert(p.HasDefault && p.DefaultValue == 2);
assert(strcmp(p.DefiningClass.Name, 'MetaBase_'));
p = props(strcmp(names, 'imm'));
assert(strcmp(p.GetAccess, 'public') && strcmp(p.SetAccess, 'immutable'));
p = props(strcmp(names, 'K'));
assert(p.Constant && strcmp(p.SetAccess, 'none') && p.DefaultValue == 6);
p = props(strcmp(names, 'dep'));
assert(p.Dependent && ~p.HasDefault);
p = props(strcmp(names, 'hid'));
assert(p.Hidden);
p = props(strcmp(names, 'tr'));
assert(p.Transient);
assert(isequal([props.Constant], strcmp(names, 'K')));

% MethodList
mnames = {mc.MethodList.Name};
assert(all(ismember({'MetaBase_', 'pubMeth', 'privMeth', 'protMeth', 'statMeth', 'hidMeth', 'empty', 'delete', 'isvalid'}, mnames)));
assert(~ismember('get.dep', mnames));
methodsList = mc.MethodList;
m = methodsList(strcmp(mnames, 'privMeth'));
assert(strcmp(m.Access, 'private') && ~m.Static);
m = methodsList(strcmp(mnames, 'statMeth'));
assert(m.Static && strcmp(m.Access, 'public'));
m = methodsList(strcmp(mnames, 'hidMeth'));
assert(m.Hidden);
m = methodsList(strcmp(mnames, 'pubMeth'));
assert(isequal(m.InputNames, {'obj'; 'a'; 'b'}) && isequal(m.OutputNames, {'r'}));
assert(strcmp(m.DefiningClass.Name, 'MetaBase_'));
m = methodsList(strcmp(mnames, 'delete'));
assert(strcmp(m.DefiningClass.Name, 'handle'));

% Subclass: inherited private members are not listed
child = MetaChild_();
mcc = metaclass(child);
assert(strcmp(mcc.Name, 'MetaChild_'));
cnames = {mcc.PropertyList.Name};
assert(isequal(sort(cnames), sort({'own', 'pub', 'prot', 'imm', 'hid', 'K', 'dep', 'tr'})));
assert(numel(mcc.SuperclassList) == 1);
assert(strcmp(mcc.SuperclassList(1).Name, 'MetaBase_'));
cm = {mcc.MethodList.Name};
assert(ismember('childMeth', cm) && ismember('protMeth', cm) && ~ismember('privMeth', cm));
assert(~ismember('MetaBase_', cm));

ma = ?MetaAbs_;
assert(ma.Abstract);
assert(~ma.HandleCompatible);
assert(isempty(ma.SuperclassList));
assert(ma.MethodList(strcmp({ma.MethodList.Name}, 'doIt')).Abstract);

% Built-in and unknown classes
md = metaclass(5);
assert(strcmp(md.Name, 'double'));
mu = ?NoSuchClassXYZ_;
assert(isempty(mu));

% Other attribute syntaxes: a single class, ~Flag, a quoted value
mf = ?MetaForms_;
fp = mf.PropertyList;
fnames = {fp.Name};
pa = fp(strcmp(fnames, 'a'));
assert(iscell(pa.GetAccess) && numel(pa.GetAccess) == 1);
assert(strcmp(pa.GetAccess{1}.Name, 'MetaChild_'));
pb = fp(strcmp(fnames, 'b'));
assert(~pb.Hidden && strcmp(pb.SetAccess, 'private') && strcmp(pb.GetAccess, 'public'));
fm = mf.MethodList(strcmp({mf.MethodList.Name}, 'm'));
assert(iscell(fm.Access) && numel(fm.Access) == 2);
f = MetaForms_();
assert(f.b == 2);
try
    x = f.a;
    error('no error');
catch e
    assert(strcmp(e.identifier, 'MATLAB:class:GetProhibited'));
end

% --- introspection respects Access and Hidden
b = MetaBase_();
visible = {'pub'; 'imm'; 'K'; 'dep'; 'tr'};
assert(isequal(properties(b), visible));
assert(isequal(fieldnames(b), visible));
assert(isequal(properties(child), [{'own'}; visible]));
% isprop is true for every property, whatever its access
assert(isprop(b, 'priv') && isprop(b, 'hid') && isprop(b, 'K') && isprop(b, 'dep'));
assert(~isprop(b, 'pubMeth'));
assert(ismethod(b, 'pubMeth') && ismethod(b, 'statMeth'));
assert(~ismethod(b, 'privMeth') && ~ismethod(b, 'protMeth') && ~ismethod(b, 'hidMeth'));
ml = methods(b);
assert(iscell(ml) && size(ml, 2) == 1);
assert(all(ismember({'MetaBase_', 'pubMeth', 'statMeth', 'delete'}, ml)));
assert(~any(ismember({'privMeth', 'protMeth', 'hidMeth', 'empty'}, ml)));
assert(isequal(sort(ml), ml));
assert(isequal(methods('MetaBase_'), ml));

disp('SUCCESS');
