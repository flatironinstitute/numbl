% Test that h(k) = class_instance works when h is an uninitialized output
% variable of a function. In MATLAB, output variables start as undefined,
% so h(1) = obj creates h as the object. The numbl codegen initialises
% output variables to 0, so the runtime must handle this case.

h = make_dog_helper_();
assert(strcmp(h.name, 'Rex'), 'h should be the Dog_ instance');

disp('SUCCESS');
