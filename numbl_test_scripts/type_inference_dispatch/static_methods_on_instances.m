% Test calling static methods on class instances (not just ClassName.method)
% In MATLAB, x.staticMethod(...) works the same as ClassName.staticMethod(...)
% when x is an instance of ClassName.

% --- Basic: call static method on an instance ---
obj = MathUtils_(5);
assert(obj.square(3) == 9);
assert(obj.double_it(7) == 14);

% --- Static method on factory-created instance ---
obj2 = MathUtils_.create(10);
assert(obj2.square(4) == 16);

% --- Chain: create instance then call static method on it ---
assert(MathUtils_.create(1).square(6) == 36);

% --- Static calling static via instance ---
obj3 = MathUtils_(2);
assert(obj3.quad(3) == 36);

% --- Mix instance and static methods on same object ---
obj4 = MathUtils_(8);
assert(obj4.get_value() == 8);       % instance method
assert(obj4.square(4) == 16);        % static method
assert(obj4.add_to_value(2) == 10);  % instance method
assert(obj4.double_it(3) == 6);      % static method

disp('SUCCESS')
