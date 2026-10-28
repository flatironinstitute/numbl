% Events and listeners of handle classes: addlistener, notify, listener
% objects, order, Enabled, delete, access attributes, event data.

L = EvLog();
s = EvSource();

% --- addlistener returns an event.listener
lh1 = addlistener(s, 'Evt1', @(src, evt) L.add(sprintf('A:%s:%s:%d', class(evt), evt.EventName, src == s)));
assert(isa(lh1, 'event.listener'));
assert(isa(lh1, 'handle'));
assert(strcmp(class(lh1), 'event.listener'));
assert(iscell(lh1.Source) && isequal(size(lh1.Source), [1 1]));
assert(lh1.Source{1} == s);
assert(strcmp(lh1.EventName, 'Evt1'));
assert(isa(lh1.Callback, 'function_handle'));
assert(lh1.Enabled == true && islogical(lh1.Enabled));
assert(lh1.Recursive == false);

% --- listeners run most recently added first, with (src, event.EventData)
lh2 = addlistener(s, 'Evt1', @(~, ~) L.add('B'));
lh3 = addlistener(s, 'Evt1', @(~, ~) L.add('C'));
s.fire('Evt1');
assert(strcmp(L.str(), 'C,B,A:event.EventData:Evt1:1'));

% --- Enabled = false skips a listener; delete removes it
L.clear();
lh1.Enabled = false;
s.fire('Evt1');
assert(strcmp(L.str(), 'C,B'));
% (MATLAB moves a re-enabled listener to the front of the order; numbl
% does not, so only membership is checked after re-enabling.)
lh1.Enabled = true;
delete(lh2);
assert(~isvalid(lh2));
assert(isvalid(lh1));
L.clear();
s.fire('Evt1');
assert(isequal(sort(L.items), {'A:event.EventData:Evt1:1', 'C'}));
expectError(@() lh2.EventName, 'MATLAB:class:InvalidHandle', 'Invalid or deleted object.');
delete(lh1);
delete(lh3);
L.clear();
s.fire('Evt1');
assert(isempty(L.items));

% --- a new listener after deleting another goes first
la = addlistener(s, 'Evt2', @(~, ~) L.add('a'));
lb = addlistener(s, 'Evt2', @(~, ~) L.add('b'));
lc = addlistener(s, 'Evt2', @(~, ~) L.add('c'));
delete(lb);
ld = addlistener(s, 'Evt2', @(~, ~) L.add('d'));
s.fire('Evt2');
assert(strcmp(L.str(), 'd,c,a'));
delete(la); delete(lc); delete(ld);

% --- NotifyAccess defaults to public: notify from outside the class
L.clear();
lh = addlistener(s, 'Evt1', @(~, ~) L.add('outside'));
notify(s, 'Evt1');
s.notify('Evt1');
assert(strcmp(L.str(), 'outside,outside'));
delete(lh);

% --- access attributes
expectError(@() notify(s, 'PrivNotify'), 'MATLAB:class:NotifyRestricted', ...
    'Cannot notify listeners of event ''PrivNotify'' in class ''EvSource''.');
expectError(@() addlistener(s, 'PrivListen', @(~, ~) 1), 'MATLAB:class:ListenRestricted', ...
    'Cannot listen to event ''PrivListen'' in class ''EvSource''.');
expectError(@() addlistener(s, 'ProtBoth', @(~, ~) 1), 'MATLAB:class:ListenRestricted', ...
    'Cannot listen to event ''ProtBoth'' in class ''EvSource''.');
expectError(@() addlistener(s, 'Nope', @(~, ~) 1), 'MATLAB:class:invalidEvent', ...
    'Event ''Nope'' is not defined for class ''EvSource''.');
expectError(@() addlistener(s, 'evt1', @(~, ~) 1), 'MATLAB:class:invalidEvent', ...
    'Event ''evt1'' is not defined for class ''EvSource''.');
expectError(@() notify(s, 'Nope'), 'MATLAB:class:invalidEvent', ...
    'Event ''Nope'' is not defined for class ''EvSource''.');
expectError(@() addlistener(s, 'Evt1', 5), 'MATLAB:class:InvalidInputArgumentTypeSizeOrValue', ...
    'Invalid input argument for function ''addlistener''.');
expectError(@() notify(s, 'Evt1', 5), 'MATLAB:class:InvalidInputArgumentTypeSizeOrValue', ...
    'Invalid input argument for function ''notify''.');
L.clear();
lp = s.listenPriv(@(~, ~) L.add('priv'));
assert(isa(lp, 'event.listener'));
delete(lp);

% --- inherited events; protected access from a subclass
c = EvChild();
lc1 = addlistener(c, 'Evt2', @(src, evt) L.add(['inherited ' class(src)]));
lc2 = addlistener(c, 'ChildEvt', @(~, evt) L.add(evt.EventName));
lc3 = c.listenProt(@(~, evt) L.add(evt.EventName));
L.clear();
c.fire('Evt2');
notify(c, 'ChildEvt');
c.fireProt();
assert(strcmp(L.str(), 'inherited EvChild,ChildEvt,ProtBoth'));
expectError(@() notify(c, 'ProtBoth'), 'MATLAB:class:NotifyRestricted', ...
    'Cannot notify listeners of event ''ProtBoth'' in class ''EvSource''.');
expectError(@() addlistener(c, 'PrivListen', @(~, ~) 1), 'MATLAB:class:ListenRestricted', ...
    'Cannot listen to event ''PrivListen'' in class ''EvSource''.');

% --- event data: a subclass of event.EventData; notify sets Source/EventName
L.clear();
lpay = addlistener(s, 'Evt2', @(src, evt) L.add(sprintf('%s:%s:%d', class(evt), evt.EventName, evt.Payload)));
d = EvPayload(42);
s.fireWith('Evt2', d);
assert(strcmp(L.str(), 'EvPayload:Evt2:42'));
assert(strcmp(d.EventName, 'Evt2'));
assert(d.Source == s);
assert(isa(d, 'event.EventData'));
delete(lpay);
ed = event.EventData();
expectError(@() setSource(ed), 'MATLAB:class:SetProhibited', '');

% --- object arrays: one listener per element
a = [EvSource(), EvSource()];
L.clear();
larr = addlistener(a, 'Evt1', @(src, ~) L.add('arr'));
assert(isa(larr, 'event.listener'));
assert(isequal(size(larr), [2 1]));
notify(a(2), 'Evt1');
assert(numel(L.items) == 1);
notify(a, 'Evt1');
assert(numel(L.items) == 3);

% --- handle equality
assert(s == s);
assert(~(s == a(1)));
assert(s ~= a(1));
assert(isequal(a == a(2), [false true]));

% --- Recursive = false: a callback that triggers its own event again
L.clear();
lr = addlistener(s, 'Evt1', @(src, ~) recurse(src, L));
s.fire('Evt1');
assert(strcmp(L.str(), 'r'));
lr.Recursive = true;
L.clear();
s.fire('Evt1');
assert(strcmp(L.str(), 'r,r,r'));
delete(lr);

% --- listeners added during a notification are not called for it;
% deleted ones are skipped
L.clear();
lx = addlistener(s, 'Evt2', @(~, ~) L.add('x'));
ladd = addlistener(s, 'Evt2', @(src, ~) addAnother(src, L));
s.fire('Evt2');
assert(strcmp(L.str(), 'adding,x'));
L.clear();
s.fire('Evt2');
assert(strcmp(L.str(), 'added,adding,x'));
delete(ladd);
L.clear();
s2 = EvSource();
lv = addlistener(s2, 'Evt2', @(~, ~) L.add('victim'));
lk = addlistener(s2, 'Evt2', @(~, ~) delete(lv));
s2.fire('Evt2');
assert(isempty(L.items));

% --- addlistener ties the listener to the source; listener() does not
s3 = EvSource();
L.clear();
addlistener(s3, 'Evt1', @(~, ~) L.add('tied'));
clear ans
s3.fire('Evt1');
assert(strcmp(L.str(), 'tied'));
lt = listener(s3, 'Evt2', @(~, ~) L.add('untied'));
assert(isa(lt, 'event.listener'));
L.clear();
s3.fire('Evt2');
assert(strcmp(L.str(), 'untied'));
clear lt
L.clear();
s3.fire('Evt2');
assert(isempty(L.items));
le = event.listener(s3, 'Evt2', @(~, ~) L.add('ctor'));
s3.fire('Evt2');
assert(strcmp(L.str(), 'ctor'));
delete(le);

% --- delete(obj): ObjectBeingDestroyed, then the delete method
t = EvSource();
L.clear();
lt2 = addlistener(t, 'ObjectBeingDestroyed', @(src, evt) L.add(sprintf('destroying %s %s %d', class(src), evt.EventName, src.Val)));
delete(t);
assert(strcmp(L.str(), 'destroying EvSource ObjectBeingDestroyed 1'));
assert(~isvalid(t));
assert(isvalid(lt2));
expectError(@() t.Val, 'MATLAB:class:InvalidHandle', 'Invalid or deleted object.');
expectError(@() notify(t, 'Evt1'), 'MATLAB:class:InvalidHandle', 'Invalid or deleted object.');
expectError(@() addlistener(t, 'Evt1', @(~, ~) 1), 'MATLAB:class:InvalidHandle', 'Invalid or deleted object.');
c2 = EvChild();
c2.Log = L;
L.clear();
lc4 = addlistener(c2, 'ObjectBeingDestroyed', @(~, ~) L.add('child destroying'));
delete(c2);
assert(strcmp(L.str(), 'child destroying,delete method'));
assert(~isvalid(c2));
assert(isequal(isvalid([s, a]), [true true true]));

% --- events() and meta.class EventList
names = events(s);
assert(isequal(names, {'Evt1'; 'Evt2'; 'PrivNotify'; 'ObjectBeingDestroyed'}));
assert(isequal(events('EvChild'), {'ChildEvt'; 'Evt1'; 'Evt2'; 'PrivNotify'; 'ObjectBeingDestroyed'}));
mc = ?EvSource;
el = mc.EventList;
assert(isequal(size(el), [7 1]));
assert(isequal({el.Name}, {'Evt1', 'Evt2', 'PrivNotify', 'PrivListen', 'ProtBoth', 'HidEvt', 'ObjectBeingDestroyed'}));
assert(strcmp(el(3).NotifyAccess, 'private') && strcmp(el(3).ListenAccess, 'public'));
assert(strcmp(el(4).ListenAccess, 'private'));
assert(el(6).Hidden);
assert(strcmp(el(7).NotifyAccess, 'private'));
assert(isa(el(1), 'meta.event'));
assert(strcmp(el(1).DefiningClass.Name, 'EvSource'));

% --- methods/ismethod include the methods inherited from handle
m = methods(s);
assert(isequal(m', {'EvSource', 'addlistener', 'delete', 'eq', 'findobj', ...
    'findprop', 'fire', 'fireWith', 'ge', 'gt', 'isvalid', 'le', ...
    'listenPriv', 'listener', 'lt', 'ne', 'notify', 'setOther'}));
assert(ismethod(s, 'notify') && ismethod(s, 'addlistener') && ismethod(s, 'isvalid'));
assert(isequal(methods('handle')', {'addlistener', 'delete', 'eq', 'findobj', ...
    'findprop', 'ge', 'gt', 'isvalid', 'le', 'listener', 'lt', 'ne', 'notify'}));

disp('SUCCESS');

function recurse(src, L)
L.add('r');
if numel(L.items) < 3
    notify(src, 'Evt1');
end
end

function addAnother(src, L)
L.add('adding');
addlistener(src, 'Evt2', @(~, ~) L.add('added'));
end

function setSource(ed)
ed.Source = 1;
end

function expectError(f, id, msg)
try
    f();
catch e
    assert(strcmp(e.identifier, id), sprintf('identifier: %s', e.identifier));
    if ~isempty(msg)
        assert(strcmp(e.message, msg), sprintf('message: %s', e.message));
    end
    return;
end
error('expected an error: %s', id);
end
