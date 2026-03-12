function r = apply_to_wrapper(wrapper, x)
    % External function that receives a Wrapper_ and calls its method
    r = wrapper.apply_inner(x);
end
