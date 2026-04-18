% Test sub2ind validates subscript bounds (MATLAB-compatible)
threw = false;
try
    sub2ind([3, 4], 5, 1);
catch
    threw = true;
end
if ~threw, error('sub2ind([3,4],5,1) should have errored'); end

threw = false;
try
    sub2ind([3, 4], 1, 10);
catch
    threw = true;
end
if ~threw, error('sub2ind([3,4],1,10) should have errored'); end

threw = false;
try
    sub2ind([3, 4], 0, 1);
catch
    threw = true;
end
if ~threw, error('sub2ind([3,4],0,1) should have errored'); end

% Trailing dims (beyond shape length) must equal 1.
threw = false;
try
    sub2ind([3, 4], 1, 1, 2);
catch
    threw = true;
end
if ~threw, error('sub2ind([3,4],1,1,2) should have errored (trailing dim > 1)'); end

% Valid calls must still succeed.
assert(sub2ind([3, 4], 2, 3) == 8);
assert(sub2ind([3, 4], 1, 1) == 1);
assert(sub2ind([3, 4], 3, 4) == 12);

disp('SUCCESS');
