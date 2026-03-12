function result = wrapper(obj, x)
% Wrapper that calls process — since this method calls process with
% runtime-typed args, it forces the dispatchUnknown path
    result = process(obj, x);
end
