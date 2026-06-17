% MATLAB's size constructors accept a trailing class-name string
% (zeros(2,3,'int32')) and the 'like' prototype form. numbl is double-only,
% so it accepts and silently ignores the class spec rather than erroring.

%% zeros / ones with a trailing class name (char and string forms)
assert(isequal(zeros(2,3,'int32'), zeros(2,3)), 'zeros(...,char)');
assert(isequal(zeros(2,3,"int32"), zeros(2,3)), 'zeros(...,string)');
assert(isequal(ones(3,'single'), ones(3)), 'ones(n,class)');
assert(isequal(zeros([2 4],'uint8'), zeros(2,4)), 'zeros([m n],class)');
assert(isequal(zeros('int32'), 0), 'zeros(class) scalar');

%% Result type is still double
assert(strcmp(class(zeros(2,'int32')), 'double'), 'still double');

%% eye / nan / inf
assert(isequal(eye(3,'int32'), eye(3)), 'eye(n,class)');
assert(isequal(eye(2,4,'double'), eye(2,4)), 'eye(m,n,class)');
assert(isequal(size(nan(2,2,'single')), [2 2]), 'nan(...,class)');
assert(isequal(size(Inf(2,'double')), [2 2]), 'Inf(...,class)');

%% rand / randn / randi
assert(isequal(size(rand(2,3,'single')), [2 3]), 'rand(...,class)');
assert(isequal(size(randn(4,'double')), [4 4]), 'randn(...,class)');
assert(isequal(size(randi(10,2,3,'int32')), [2 3]), 'randi(...,class)');
assert(isscalar(randi(10,'int32')), 'randi(imax,class) scalar');

%% true / false (logical class)
assert(islogical(true(2,'logical')), 'true(...,class)');
assert(isequal(size(false(2,3,'logical')), [2 3]), 'false(...,class)');

%% 'like' prototype form is accepted and ignored
proto = 1;
assert(isequal(zeros(2,3,'like',proto), zeros(2,3)), 'zeros(...,like,proto)');
assert(isequal(size(ones([2 2],'like',proto)), [2 2]), 'ones(...,like,proto)');

disp('SUCCESS');
