% Test that indexed assignment into a class instance variable works
% e.g., f(k) = tweakDomain(f(k), ...) where f is a class instance

% Simple class that holds a value
% We'll use a struct to simulate since we just need indexed assignment on a scalar

% Test: assign class instance back via f(1) = value
MyObj = struct('x', 10, 'y', 20);
MyObj(1) = struct('x', 30, 'y', 40);
assert(MyObj.x == 30, 'indexed assignment f(1) = struct should work');
assert(MyObj.y == 40);

disp('SUCCESS');
