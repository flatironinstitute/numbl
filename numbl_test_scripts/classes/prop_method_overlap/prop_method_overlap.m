% Test: folder-based class where a separate method file has the same name
% as a property. MATLAB allows this (silently ignoring the method) when the
% method is defined in a separate file inside an @-folder, but errors when
% the conflicting method is defined inline in the classdef file.
%
% Real-world example: chebfun's @adchebfun class has a property "domain"
% and a separate file "domain.m" method:
% https://github.com/chebfun/chebfun/tree/0b2c73e34720f05ac4c4f52ef31bed78c3e0fbb0/%40adchebfun
%
% PropMethodClass has a property "domain" and a separate @-folder method
% file "domain.m". The property should take precedence and the method
% should be silently ignored.

obj = PropMethodClass([1, 2, 3]);

% Accessing .domain should return the property value, not call the method
result = obj.domain;
assert(isequal(result, [1, 2, 3]));

disp('SUCCESS')
