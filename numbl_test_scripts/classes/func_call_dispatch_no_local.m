% Test function-call syntax dispatching to class method
% when there is no local function with the same name.
% MATLAB rule: if no nested/local function exists, check if the
% first argument is a class instance with a matching method.

X = FuncCallDispatchTarget_(5);

% Function-call syntax should dispatch to class method
Y = add_to_value(X, 2);
assert(Y.value == 7);

% Dot syntax should also work
Z = X.add_to_value(10);
assert(Z.value == 15);

disp('SUCCESS')
