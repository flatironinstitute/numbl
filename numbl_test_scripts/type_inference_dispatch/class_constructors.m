% Test class constructors: hard scenarios including cross-class construction,
% constructor calling methods, objects in loops, conditional construction

% --- Basic constructor with defaults ---
c = SimpleCounter_();
% c is reassigned in a loop (line 67-68) and at line 95, but all
% assignments are ClassInstance<SimpleCounter_> so the type is preserved.
assert(strcmp(__inferred_type_str(c), 'ClassInstance<SimpleCounter_>'));
assert(c.Value == 0);
assert(c.Step == 1);

c2 = SimpleCounter_(10, 3);
assert(strcmp(__inferred_type_str(c2), 'ClassInstance<SimpleCounter_>'));
assert(c2.Value == 10);
assert(c2.Step == 3);

% --- Method that calls increment in a loop ---
c3 = SimpleCounter_(0, 5);
c3 = c3.increment_n(4);
assert(c3.Value == 20, 'increment_n calls increment 4 times');

% --- Cross-class construction: method returns instance of different class ---
c4 = SimpleCounter_(100);
assert(strcmp(__inferred_type_str(c4), 'ClassInstance<SimpleCounter_>'));
cfg = c4.to_configured();
assert(strcmp(__inferred_type_str(cfg), 'ClassInstance<ConfiguredObj_>'));
assert(cfg.Speed == 100, 'high value -> fast mode');
assert(strcmp(cfg.Mode, 'fast'));

c5 = SimpleCounter_(10);
cfg2 = c5.to_configured();
assert(strcmp(__inferred_type_str(cfg2), 'ClassInstance<ConfiguredObj_>'));
assert(cfg2.Speed == 30, 'low value -> quality mode');
assert(strcmp(cfg2.Mode, 'quality'));

% --- Call method on cross-class result (chained type inference) ---
assert(c4.to_configured().describe() == 150);
assert(c5.to_configured().describe() == 130);

% --- Reverse direction: ConfiguredObj_ returns SimpleCounter_ ---
cfg3 = ConfiguredObj_('fast');
assert(strcmp(__inferred_type_str(cfg3), 'ClassInstance<ConfiguredObj_>'));
counter_back = cfg3.to_counter();
% counter_back is reassigned below (increment), so skip type assert here
assert(counter_back.Value == 100);
assert(counter_back.Step == 50);
counter_back = counter_back.increment();
assert(counter_back.Value == 150);

% --- Method taking another instance of same class ---
cfg4 = ConfiguredObj_('fast');
assert(strcmp(__inferred_type_str(cfg4), 'ClassInstance<ConfiguredObj_>'));
cfg5 = ConfiguredObj_('quality');
assert(strcmp(__inferred_type_str(cfg5), 'ClassInstance<ConfiguredObj_>'));
combined = cfg4.combine(cfg5);
assert(strcmp(__inferred_type_str(combined), 'ClassInstance<ConfiguredObj_>'));
assert(strcmp(combined.Mode, 'balanced'), '100+30=130 <= 150');

cfg6 = ConfiguredObj_('fast');
combined2 = cfg6.combine(cfg6);
assert(strcmp(__inferred_type_str(combined2), 'ClassInstance<ConfiguredObj_>'));
assert(strcmp(combined2.Mode, 'fast'), '100+100=200 > 150');

% --- Object creation inside a loop, accumulate results ---
total = 0;
for i = 1:5
    assert(strcmp(__inferred_type_str(i), "Number"));
    c = SimpleCounter_(i * 10, i);
    c = c.increment();
    total = total + c.Value;
end
% i=1: val=10, step=1, after inc=11
% i=2: val=20, step=2, after inc=22
% i=3: val=30, step=3, after inc=33
% i=4: val=40, step=4, after inc=44
% i=5: val=50, step=5, after inc=55
assert(total == 165);

% --- Conditional construction ---
for i = 1:2
    if i == 1
        obj = SimpleCounter_(99);
    else
        obj = ConfiguredObj_('fast');
    end
    % obj is assigned different classes in each branch, so type is Unknown
    assert(strcmp(__inferred_type_str(obj), 'Unknown'));
    if i == 2
        assert(obj.describe() == 150);
    end
end

% --- Object equality method: passing same-class instance ---
a = SimpleCounter_(5, 2);
b = SimpleCounter_(5, 2);
c = SimpleCounter_(5, 3);
assert(a.equals(b));
assert(~a.equals(c));

% --- Constructor result used directly in expression ---
assert(SimpleCounter_(42).Value == 42);
assert(ConfiguredObj_('fast').Speed == 100);

disp('SUCCESS')
