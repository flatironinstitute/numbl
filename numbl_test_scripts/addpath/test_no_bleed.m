% Test that paths from other tests do not bleed into this test.
% If addpath state leaked between tests, helper_a/helper_b might be available.

try
    helper_a(1);
    error('helper_a should NOT be available — path bleed detected');
catch
end

try
    helper_b(1);
    error('helper_b should NOT be available — path bleed detected');
catch
end

try
    shared_func(1);
    error('shared_func should NOT be available — path bleed detected');
catch
end

disp('SUCCESS');
