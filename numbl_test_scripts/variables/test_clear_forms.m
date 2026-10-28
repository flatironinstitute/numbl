% What each form of `clear` removes: variables, links to globals, globals,
% persistents. Helpers: clear_counter_.m (persistent counter) and
% clear_getg_.m (reads global G).

% clear keeps a global's value and removes only the link
global G
G = 5;
clear
assert(exist('G', 'var') == 0);
assert(clear_getg_() == 5);
assert(isequal(who('global'), {'G'}));
global G
assert(G == 5);

% clear NAME on a global removes only the link
clear G
assert(exist('G', 'var') == 0);
assert(clear_getg_() == 5);

% clear global removes the globals
global G
G = 7;
clear global
assert(exist('G', 'var') == 0);
assert(isempty(clear_getg_()));

% clear global NAME removes that global only
global H K
H = 1; K = 2;
clear global H
assert(exist('H', 'var') == 0);
assert(exist('K', 'var') == 1);
assert(isequal(sort(who('global')), {'G'; 'K'}));

% persistents survive clear, not clear functions or clear NAME
clear_counter_(); clear_counter_();
clear
assert(clear_counter_() == 3);
clear functions
assert(clear_counter_() == 1);
clear_counter_();
clear clear_counter_
assert(clear_counter_() == 1);

% clear all removes variables, globals and persistents
global Z
Z = 3;
x = 1;
clear all
assert(clear_counter_() == 1);
assert(exist('x', 'var') == 0);
global Z
assert(isempty(Z));

% clear functions keeps variables and globals
global Q
Q = 9; y = 2;
clear functions
assert(exist('y', 'var') == 1);
assert(Q == 9);

% clear variables keeps the globals' values
clear variables
global Q
assert(Q == 9);

% the PRNG stream, warning state, appdata and tic survive clear all
rng(3); rand; clear all; b = rand; rng(3); rand; assert(b == rand);
warning('off', 'my:id');
clear all
s = warning('query', 'my:id');
assert(strcmp(s.state, 'off'));
warning('on', 'my:id');
setappdata(0, 'k', 5);
clear all
assert(getappdata(0, 'k') == 5);
rmappdata(0, 'k');

clear global
disp('SUCCESS')
