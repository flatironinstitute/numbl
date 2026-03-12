classdef HostClass
    properties
        vals
    end
    methods
        function obj = HostClass(v)
            obj.vals = v;
        end
        function result = InfoClass(obj, flag)
            % This method has the same name as the InfoClass constructor.
            % When called as InfoClass(hostObj, 'get'), MATLAB should
            % dispatch to this method, not the InfoClass constructor.
            if strcmp(flag, 'get')
                result = obj.vals;
            else
                result = [];
            end
        end
    end
end
