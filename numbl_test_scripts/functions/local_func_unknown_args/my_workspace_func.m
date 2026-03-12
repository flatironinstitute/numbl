function result = my_workspace_func(n, x)
    c = {n, x};
    result = my_helper(c{1}, c{2});
end

function y = my_helper(s, x)
    y = x * s;
end
