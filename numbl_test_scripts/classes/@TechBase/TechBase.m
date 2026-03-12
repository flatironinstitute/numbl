classdef TechBase
    properties
        data
    end
    methods
        function obj = TechBase(op, data, pref)
            if nargin < 3 || isempty(pref)
                pref = TechBase.techPref();
            else
                pref = TechBase.techPref(pref);
            end
            obj.data = pref;
        end
        function f = doCompose(f, op, data, pref)
            % Mirrors chebtech/compose.m pattern:
            % 1. Call static method via instance
            if nargin < 4
                pref = f.techPref();
            else
                pref = f.techPref(pref);
            end
            % 2. Call make via instance (like f.make(op, data, pref))
            f = f.make(op, data, pref);
        end
    end
    methods (Access = public, Static = true)
        outPref = techPref(inPref)
    end
end
