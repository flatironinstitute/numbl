classdef JitLocal
    properties
        val
    end
    methods
        function obj = JitLocal(v)
            obj.val = v;
        end
    end
    methods
        result = compute(obj, op, data)
    end
end
