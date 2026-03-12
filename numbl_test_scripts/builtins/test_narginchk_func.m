% Test narginchk and nargoutchk builtins

function test_narginchk_func()
    test_basic(1, 2, 3);
    test_basic(1, 2);

    ok = false;
    try
        test_basic(1);
    catch
        ok = true;
    end
    assert(ok, 'Should have thrown for too few args');

    ok2 = false;
    try
        test_basic(1, 2, 3, 4);
    catch
        ok2 = true;
    end
    assert(ok2, 'Should have thrown for too many args');

    disp('SUCCESS');
end

function test_basic(a, b, c, d)
    narginchk(2, 3);
end
