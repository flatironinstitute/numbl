function result = process(obj, x)
% Primary method: calls a local helper that receives the class instance
    result = helperExtract(obj, x);
end

function val = helperExtract(obj, x)
% Local helper — receives a class instance as first arg
    val = obj.value * x;
end
