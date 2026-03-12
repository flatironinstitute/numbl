% Test direct property assignment on handle objects

h = NumHandle_();

% Direct property assignment
h.Number = 42;
assert(h.Number == 42, 'direct property assignment');

% Multiple handles, direct assignment through one
h2 = h;
h2.Number = 100;
assert(h.Number == 100, 'assignment through second handle visible on first');

% Reassign variable to a new object - should NOT affect the other handle
h3 = NumHandle_();
h3.Number = 999;
h4 = h3;
h3 = NumHandle_();  % h3 now points to a new object
h3.Number = 1;
assert(h4.Number == 999, 'reassigning h3 should not affect h4');
assert(h3.Number == 1, 'h3 is a new independent object');

% Multiple independent handle objects
a = NumHandle_();
b = NumHandle_();
a.Number = 10;
b.Number = 20;
assert(a.Number == 10, 'independent handle objects');
assert(b.Number == 20, 'independent handle objects');

fprintf('SUCCESS\n');
