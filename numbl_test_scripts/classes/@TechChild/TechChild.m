classdef TechChild < TechBase
    methods
        function obj = TechChild(op, data, pref)
            if nargin < 3 || isempty(pref)
                pref = TechBase.techPref();
            else
                pref = TechBase.techPref(pref);
            end
            obj.data = pref;
        end
    end
    methods (Access = public, Static = true)
        f = make(varargin)
    end
end
