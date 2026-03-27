% Test assert(NaN) should fail
% MATLAB: NaN is not truthy, assert(NaN) throws an error

passed = false;
try
    assert(NaN);
    passed = true;
catch
end
assert(~passed, 'assert(NaN) should throw an error');

% assert(0) should also fail
passed2 = false;
try
    assert(0);
    passed2 = true;
catch
end
assert(~passed2, 'assert(0) should throw an error');

% assert with NaN in tensor should also fail
passed3 = false;
try
    assert([1 NaN 1]);
    passed3 = true;
catch
end
assert(~passed3, 'assert([1 NaN 1]) should throw an error');

% assert(1) should pass
assert(1);
assert(true);

disp('SUCCESS');
