% Property events: PreSet/PostSet listeners on SetObservable properties and
% PreGet/PostGet on GetObservable ones.

L = EvLog();
s = EvSource();

% --- PostSet and PreSet listeners: event.proplistener
p1 = addlistener(s, 'Val', 'PostSet', @(src, evt) L.add(sprintf('post %s %s %s %d %d', ...
    class(src), class(evt), evt.EventName, evt.AffectedObject == s, evt.AffectedObject.Val)));
assert(isa(p1, 'event.proplistener'));
assert(isa(p1, 'event.listener'));
assert(iscell(p1.Object) && p1.Object{1} == s);
assert(iscell(p1.Source) && isa(p1.Source{1}, 'meta.property'));
assert(strcmp(p1.Source{1}.Name, 'Val'));
assert(strcmp(p1.EventName, 'PostSet'));
p2 = addlistener(s, 'Val', 'PreSet', @(src, evt) L.add(sprintf('pre %s %s %d', src.Name, evt.EventName, evt.AffectedObject.Val)));
s.Val = 7;
assert(strcmp(L.str(), 'pre Val PreSet 1,post matlab.metadata.Property event.PropertyEvent PostSet 1 7'));
% Both fire even when the value does not change.
L.clear();
s.Val = 7;
assert(strcmp(L.str(), 'pre Val PreSet 7,post matlab.metadata.Property event.PropertyEvent PostSet 1 7'));
% The meta.property is the one in the class's PropertyList.
mc = ?EvSource;
pl = mc.PropertyList;
assert(pl(strcmp({pl.Name}, 'Val')) == p1.Source{1});
assert(pl(strcmp({pl.Name}, 'Val')).SetObservable);
assert(pl(strcmp({pl.Name}, 'Val')).GetObservable);
assert(~pl(strcmp({pl.Name}, 'Plain')).SetObservable);
delete(p2);
delete(p1);

% --- indexed assignment fires PostSet
L.clear();
p5 = addlistener(s, 'Val', 'PostSet', @(~, evt) L.add(mat2str(evt.AffectedObject.Val)));
s.Val(3) = 5;
assert(isequal(s.Val, [7 0 5]));
assert(strcmp(L.str(), '[7 0 5]'));
delete(p5);

% --- PreGet / PostGet
L.clear();
g1 = addlistener(s, 'Val', 'PreGet', @(~, evt) L.add(evt.EventName));
g2 = addlistener(s, 'Val', 'PostGet', @(~, evt) L.add(evt.EventName));
x = s.Val;
assert(isequal(x, [7 0 5]));
assert(strcmp(L.str(), 'PreGet,PostGet'));
delete(g1); delete(g2);

% --- assignments inside class methods fire too
L.clear();
p3 = addlistener(s, 'Other', 'PostSet', @(~, evt) L.add(sprintf('other %d', evt.AffectedObject.Other)));
s.setOther(3);
assert(strcmp(L.str(), 'other 3'));
delete(p3);

% --- one listener for several properties
L.clear();
p4 = addlistener(s, {'Val', 'Other'}, 'PostSet', @(src, ~) L.add(src.Name));
assert(isequal(size(p4.Source), [2 1]));
s.Val = 1;
s.Other = 2;
assert(strcmp(L.str(), 'Val,Other'));
delete(p4);

% --- a set method runs between PreSet and PostSet, once
c = EvChild();
c.Log = L;
L.clear();
q1 = addlistener(c, 'Clamped', 'PreSet', @(~, ~) L.add('PreSet'));
q2 = addlistener(c, 'Clamped', 'PostSet', @(~, evt) L.add(sprintf('PostSet %d', evt.AffectedObject.Clamped)));
c.Clamped = 50;
assert(c.Clamped == 10);
assert(strcmp(L.str(), 'PreSet,set.Clamped,PostSet 10'));
% inherited observable property
L.clear();
q3 = addlistener(c, 'Val', 'PostSet', @(~, ~) L.add('child Val'));
c.Val = 2;
assert(strcmp(L.str(), 'child Val'));

% --- listener on a property that is not observable
expectError(@() addlistener(s, 'Plain', 'PostSet', @(~, ~) 1), 'MATLAB:class:nonSetObservableProp', ...
    'While adding a PostSet listener, property ''Plain'' in class ''EvSource'' is not defined to be SetObservable.');
expectError(@() addlistener(s, 'Other', 'PostGet', @(~, ~) 1), 'MATLAB:class:nonGetObservableProp', ...
    'While adding a PostGet listener, property ''Other'' in class ''EvSource'' is not defined to be GetObservable.');
expectError(@() addlistener(s, 'Nope', 'PostSet', @(~, ~) 1), 'MATLAB:class:InvalidProperty', ...
    'The name ''Nope'' is not an accessible property for an instance of class ''EvSource''.');

% --- a listener callback that sets the property it observes does not
% run again for its own assignment
L.clear();
r = addlistener(s, 'Other', 'PostSet', @(~, evt) bump(evt.AffectedObject, L));
s.Other = 1;
assert(strcmp(L.str(), 'bump'));
assert(s.Other == 2);
delete(r);

disp('SUCCESS');

function bump(obj, L)
L.add('bump');
obj.Other = obj.Other + 1;
end

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
