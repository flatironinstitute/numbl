function result = refine(op, pref)
    % Static method in external file.
    % When called via obj.refine(op, pref), MATLAB does NOT pass obj
    % as the first argument because this is a static method.
    result = op + pref.value;
end
