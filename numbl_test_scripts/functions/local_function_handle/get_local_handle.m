function h = get_local_handle(scale)
    h = @(x) helper(x, scale);
end

function y = helper(x, s)
    y = x * s;
end
