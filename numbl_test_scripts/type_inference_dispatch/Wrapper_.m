classdef Wrapper_
    properties
        Inner
        Label
    end
    methods
        function obj = Wrapper_(inner, label)
            obj.Inner = inner;
            obj.Label = label;
        end
        function r = get_inner(obj)
            r = obj.Inner;
        end
        function r = apply_inner(obj, x)
            % Call method on the wrapped object
            r = obj.Inner.apply_op(x);
        end
    end
end
