function test_varargout_fixed_outputs()
  [x, y, z] = foo();
  assert(x == 1);
  assert(y == 2);
  assert(z == 3);
  disp('SUCCESS');
end

function [a, varargout] = foo()
  a = 1;
  varargout{1} = 2;
  varargout{2} = 3;
end
