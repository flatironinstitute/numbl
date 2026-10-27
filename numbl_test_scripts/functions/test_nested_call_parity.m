% Nested functions follow the same call rules as other functions.
run_checks();
disp('SUCCESS')

function run_checks()
    k = 2;
    % Called as a statement, the first output becomes ans.
    five();
    assert(ans == 5);
    % Requesting more outputs than declared is an error.
    try
        [a, b] = five(); %#ok<ASGLU>
        error('expected an error');
    catch e
        assert(contains(e.message, 'Too many output arguments'));
    end
    % Nested functions see the parent's variables.
    assert(scaled(3) == 6);

    function r = five()
        r = 5;
    end

    function r = scaled(x)
        r = x * k;
    end
end
