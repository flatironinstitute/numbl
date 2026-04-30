function [a, b] = do_fit_multi(x)
    function [p, q] = inner()
        p = scale_local(x, 2);
        q = scale_local(x, 3);
    end
    [a, b] = inner();
end

function r = scale_local(x, s)
    r = x * s;
end
