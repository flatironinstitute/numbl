function r = nr_caller()
    % A function file with a local and a nested function, next to private/.
    r.existPriv = exist('nr_priv');
    r.existPrivFile = exist('nr_priv', 'file');
    r.whichPrivFound = ~isempty(which('nr_priv'));
    r.existLocal = exist('nr_caller_loc');
    r.whichLocalFound = ~isempty(which('nr_caller_loc'));
    r.existNested = exist('nr_nested');
    r.whichNestedFound = ~isempty(which('nr_nested'));
    r.narginNested = nargin('nr_nested');
    r.narginLocal = nargin('nr_caller_loc');
    r.callNested = nr_nested(1, 2);
    h = @nr_nested;
    r.handleNested = h(3, 4);
    % A string-named call does not see nested functions.
    try
        feval('nr_nested', 1, 2);
        r.fevalNestedFails = false;
    catch
        r.fevalNestedFails = true;
    end
    r.fevalLocal = feval('nr_caller_loc', 7);
    function y = nr_nested(a, b)
        y = a + b;
    end
end

function y = nr_caller_loc(x)
    y = -x;
end
