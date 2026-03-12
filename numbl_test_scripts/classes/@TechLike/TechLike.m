classdef TechLike
    properties
        data
    end
    methods
        function obj = TechLike(d)
            if nargin > 0
                obj.data = d;
            else
                obj.data = 0;
            end
        end
        function r = callRefine(obj, op, pref)
            % Call a static method via an instance — MATLAB does NOT
            % prepend the instance as a hidden first argument for static
            % methods, so refine should receive (op, pref), not (obj, op, pref).
            r = obj.refine(op, pref);
        end
    end
    methods (Static = true)
        result = refine(op, pref)
    end
end
