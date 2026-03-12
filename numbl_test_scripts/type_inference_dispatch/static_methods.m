% Test static methods: hard scenarios - static calling static,
% static factory -> instance methods, instance calling static,
% static creating multiple instances, chained factory calls

% --- Static calling static ---
assert(MathUtils_.square(5) == 25);
assert(MathUtils_.quad(3) == 36);

% --- Static factory -> instance method ---
obj = MathUtils_.create(10);
assert(strcmp(__inferred_type_str(obj), 'ClassInstance<MathUtils_>'));
assert(obj.get_value() == 10);
assert(obj.add_to_value(5) == 15);

% --- Chain: factory result -> instance method directly ---
assert(MathUtils_.create(42).get_value() == 42);
assert(MathUtils_.create(7).add_to_value(3) == 10);

% --- Instance method calling static method ---
obj2 = MathUtils_(6);
assert(strcmp(__inferred_type_str(obj2), 'ClassInstance<MathUtils_>'));
assert(obj2.apply_static() == 36);

% --- Static method creating instances and calling their methods ---
assert(MathUtils_.sum_values(10, 20) == 30);

% --- Mix static and instance in one expression ---
r = MathUtils_.square(MathUtils_.create(5).get_value());
assert(r == 25);

% --- Static result as constructor arg ---
obj3 = MathUtils_(MathUtils_.square(3));
assert(strcmp(__inferred_type_str(obj3), 'ClassInstance<MathUtils_>'));
assert(obj3.get_value() == 9);

% --- Chained: create -> apply_static (instance calls static) ---
assert(MathUtils_.create(4).apply_static() == 16);

% --- Multiple factories in expression ---
r2 = MathUtils_.create(10).get_value() + MathUtils_.create(20).get_value();
assert(r2 == 30);

% --- Loop with static factory ---
total = 0;
for i = 1:5
    assert(strcmp(__inferred_type_str(i), "Number"));
    obj = MathUtils_.create(i * 10);
    total = total + obj.apply_static();
end
assert(total == 5500, '100+400+900+1600+2500');

disp('SUCCESS')
