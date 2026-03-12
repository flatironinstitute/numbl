function result = execute_op(obj, op)
% External method that receives a function handle and calls it
result = feval(op, obj.val);
end
