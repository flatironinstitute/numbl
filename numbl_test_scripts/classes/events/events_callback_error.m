% An error in a listener callback is reported as a warning ("Error occurred
% while executing the listener callback for event ...") and the remaining
% listeners and the code after notify still run.

L = EvLog();
s = EvSource();
l1 = addlistener(s, 'Evt2', @(~, ~) L.add('first added'));
l2 = addlistener(s, 'Evt2', @(~, ~) error('my:id', 'boom %d', 3));
l3 = addlistener(s, 'Evt2', @(~, ~) L.add('last added'));
s.fireWith('Evt2', EvPayload(7));
assert(strcmp(L.str(), 'last added,first added'));

% The same for property events.
L.clear();
p = addlistener(s, 'Val', 'PostSet', @(~, ~) error('bad listener'));
s.Val = 3;
assert(s.Val == 3);

% The warning can be turned off by its identifier.
warning('off', 'MATLAB:callback:error');
s.fire('Evt2');
warning('on', 'MATLAB:callback:error');
assert(strcmp(L.str(), 'last added,first added'));

disp('SUCCESS');
