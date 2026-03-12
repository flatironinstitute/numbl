% Test value class semantics vs handle class semantics

% === Value class: copies are independent ===
a = NumValue_();
assert(a.Number == 1, 'default value');
b = a;
a.Number = 7;
assert(a.Number == 7, 'value class: a should be 7');
assert(b.Number == 1, 'value class: b should still be 1 (independent copy)');

% === Handle class: copies share the same object ===
c = NumHandle_();
assert(c.Number == 1, 'default value');
d = c;
c.Number = 7;
assert(c.Number == 7, 'handle class: c should be 7');
assert(d.Number == 7, 'handle class: d should also be 7 (shared reference)');

% Mutate via the other handle
d.Number = 42;
assert(c.Number == 42, 'handle class: c should reflect d mutation');

% === Value class: function cannot mutate caller's copy ===
v = NumValue_();
v2 = set_number_value(v, 99);
assert(v.Number == 1, 'value class: original unchanged after function');
assert(v2.Number == 99, 'value class: returned copy has new value');

% === Handle class: function mutates caller's object ===
h = NumHandle_();
set_number_handle(h, 99);
assert(h.Number == 99, 'handle class: function should mutate caller object');

fprintf('SUCCESS\n');

function obj = set_number_value(obj, val)
    obj.Number = val;
end

function set_number_handle(obj, val)
    obj.Number = val;
end
