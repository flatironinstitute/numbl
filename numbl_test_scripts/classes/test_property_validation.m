% Classdef property validation: size/class/validator declarations convert
% and check default values and assigned values. Verified against MATLAB
% R2025b.

o = PropValid_;
% defaults: a given default is converted to the declared size; a declared
% numeric or logical property without a default gets zeros/false
assert(o.a == 1);
assert(strcmp(o.b, 'x'));
assert(isa(o.c, 'double') && isequal(size(o.c), [0 0]));
assert(isequal(o.d, [0 0 0]));
assert(isa(o.e, 'double') && isequal(o.e, 0));
assert(islogical(o.h) && isequal(o.h, false));
assert(isequal(size(o.k), [0 0]));
assert(isequal(o.m, zeros(2)));
assert(isa(o.r, 'double') && isequal(size(o.r), [1 0]));
assert(strcmp(o.color, 'red'));
assert(isa(o.s, 'double') && isequal(o.s, [0 0]));
assert(ischar(o.u) && isequal(double(o.u), 0));
assert(isstring(o.w) && o.w == "");
assert(iscell(o.z) && isequal(size(o.z), [1 0]));

% assignments are converted
o.a = 'x';
assert(isa(o.a, 'double') && o.a == 120);
o.b = "hey";
assert(ischar(o.b) && strcmp(o.b, 'hey'));
o.c = true;
assert(isa(o.c, 'double') && o.c == 1);
o.d = 7;
assert(isequal(o.d, [7 7 7]));
o.r = [1; 2];
assert(isequal(o.r, [1 2]));
o.color = 'blue';
assert(strcmp(o.color, 'blue'));
o = o.bump();
o = o.bump();
assert(o.count == 2);

% invalid assignments raise MATLAB's errors
check_err(@() setp(o, 'a', -1), 'MATLAB:validators:mustBePositive', ...
  'Error setting property ''a'' of class ''PropValid_''. Value must be positive.');
check_err(@() setp(o, 'a', [1 2]), 'MATLAB:validation:IncompatibleSize', ...
  'Error setting property ''a'' of class ''PropValid_''. Value must be a scalar.');
check_err(@() setp(o, 'a', {1}), 'MATLAB:validation:UnableToConvert', ...
  'Error setting property ''a'' of class ''PropValid_''. Value must be of type double or be convertible to double.');
check_err(@() setp(o, 'k', 'x'), 'MATLAB:validators:mustBeNumeric', ...
  'Error setting property ''k'' of class ''PropValid_''. Value must be numeric.');
check_err(@() setp(o, 'count', 1.5), 'MATLAB:validators:mustBeInteger', ...
  'Error setting property ''count'' of class ''PropValid_''. Value must be integer.');
check_err(@() setp(o, 'color', 'green'), 'MATLAB:validators:mustBeMember', ...
  sprintf('Error setting property ''color'' of class ''PropValid_''. Value must be a member of this set:\n    ''red''\n    ''blue''\n'));
% the failed assignment left the property unchanged
assert(o.a == 120);

% handle class: in the constructor and on later assignment
check_err(@() PropValidHandle_(-2), 'MATLAB:validators:mustBePositive', ...
  'Error setting property ''a'' of class ''PropValidHandle_''. Value must be positive.');
h = PropValidHandle_(2);
check_err(@() setp(h, 'a', 0), 'MATLAB:validators:mustBePositive', ...
  'Error setting property ''a'' of class ''PropValidHandle_''. Value must be positive.');
assert(h.a == 2);
h.a = int8(5);
assert(isa(h.a, 'double') && h.a == 5);

disp('SUCCESS')

function o = setp(o, name, v)
  o.(name) = v;
end

function check_err(f, id, msg)
  try
    f();
  catch e
    assert(strcmp(e.identifier, id), sprintf('identifier: got [%s], expected [%s]', e.identifier, id));
    assert(strcmp(e.message, msg), sprintf('message: got [%s], expected [%s]', e.message, msg));
    return;
  end
  error('expected an error: %s', msg);
end
