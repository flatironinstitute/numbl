% Test: varargout with more elements than nargout
% When a function sets varargout to a cell with N elements,
% but the caller only requests M < N outputs, only the first M should be returned.

function test_varargout_cell_assign3()
    % Request 1 output from a function that sets varargout to 5 elements
    a = foo();
    assert(a == 10);

    % Request 2 outputs
    [a, b] = foo();
    assert(a == 10);
    assert(b == 20);

    % Request all 5
    [a, b, c, d, e] = foo();
    assert(a == 10);
    assert(b == 20);
    assert(c == 30);
    assert(d == 40);
    assert(e == 50);

    disp('SUCCESS');
end

function varargout = foo()
    h1 = 10;
    h2 = 20;
    h3 = 30;
    h4 = 40;
    h5 = 50;
    if nargout > 0
        varargout = {h1 ; h2 ; h3 ; h4 ; h5};
    end
end
