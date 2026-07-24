% containers.Map keys() and values() return entries in key-sorted order
% (MATLAB semantics), and values() unwraps 'any'-type storage.

m = containers.Map();
m('Gradient') = 10;
m('Adc') = 'text';
m('Name') = 20;
m('Block') = 30;

k = keys(m);
assert(isequal(k, {'Adc', 'Block', 'Gradient', 'Name'}));

% values() corresponds to sorted keys and unwraps each stored value
v = values(m);
assert(numel(v) == 4);
assert(ischar(v{1}) && strcmp(v{1}, 'text'));   % Adc
assert(v{2} == 30);                             % Block
assert(v{3} == 10);                             % Gradient
assert(v{4} == 20);                             % Name

% Typed char->double map
m2 = containers.Map('KeyType', 'char', 'ValueType', 'double');
m2('zz') = 1;
m2('aa') = 2;
m2('mm') = 3;
assert(isequal(keys(m2), {'aa', 'mm', 'zz'}));
assert(isequal(cell2mat(values(m2)), [2 3 1]));

% Numeric keys sort numerically
m3 = containers.Map('KeyType', 'double', 'ValueType', 'char');
m3(10) = 'ten';
m3(2) = 'two';
m3(33) = 'thirty-three';
k3 = keys(m3);
assert(k3{1} == 2 && k3{2} == 10 && k3{3} == 33);
v3 = values(m3);
assert(strcmp(v3{1}, 'two'));
assert(strcmp(v3{2}, 'ten'));
assert(strcmp(v3{3}, 'thirty-three'));

disp('SUCCESS');
