function result = apply_op(obj)
% External method that creates a handle to a local function and passes it
% to another method
result = execute_op(obj, @my_local_func);
end

function y = my_local_func(x)
y = x * 3;
end
