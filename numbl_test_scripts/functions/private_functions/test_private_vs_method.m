% Test: private function takes precedence over class method
% In MATLAB, private functions (#6) have higher precedence than object functions (#7)
% A private function with the same name as a class method should win

obj = PrecedenceClass(10);
r = compete(obj);
assert(r == -1, 'private function should shadow class method');

disp('SUCCESS');
