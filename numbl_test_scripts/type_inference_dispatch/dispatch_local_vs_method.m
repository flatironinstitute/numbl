% Test dispatch precedence: local functions vs class methods vs file functions
% When using function-call syntax, local functions should have highest priority.
% When using dot syntax, class methods should always be called.

obj = DispTarget_(10);
assert(strcmp(__inferred_type_str(obj), 'ClassInstance<DispTarget_>'));

% Dot syntax always calls class method
assert(obj.transform(5) == 50, 'dot syntax -> class method');
assert(obj.compute() == 1010, 'dot syntax -> class method');

% Function-call syntax: local function shadows class method
assert(transform(obj, 5) == -1, 'function-call syntax -> local function');

% Function-call syntax: no local shadow -> class method dispatch
assert(compute(obj) == 1010, 'function-call syntax -> class method (no local shadow)');

% Verify with different object
obj2 = DispTarget_(20);
assert(strcmp(__inferred_type_str(obj2), 'ClassInstance<DispTarget_>'));
assert(obj2.transform(3) == 60);
assert(transform(obj2, 3) == -1, 'local still shadows');
assert(compute(obj2) == 1020);

% Use class method result in expression with local function result
r = obj.transform(5) + transform(obj, 5);
assert(r == 49, '50 + (-1) = 49');

disp('SUCCESS')

function r = transform(~, ~)
    % Local function that shadows class method name
    r = -1;
end
