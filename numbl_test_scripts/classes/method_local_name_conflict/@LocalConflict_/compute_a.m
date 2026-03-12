function result = compute_a(obj)
% compute_a: calls local helper "localHelper" which returns value + 10
result = localHelper(obj.value);
end

function y = localHelper(x)
y = x + 10;
end
