function y = apply_handle(h, x)
    % Mimics chunkerfunc receiving @splinefunc and invoking it from a different
    % scope than where the nested function was defined.
    y = h(x);
end
