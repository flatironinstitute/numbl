classdef MyErr_ < MException
    properties
        Extra
    end
    methods
        function obj = MyErr_(id, msg, extra)
            obj@MException(id, msg);
            obj.Extra = extra;
        end
        function s = describe(obj)
            s = sprintf('%s [%d]', obj.message, obj.Extra);
        end
    end
end
