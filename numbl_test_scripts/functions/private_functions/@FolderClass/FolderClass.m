classdef FolderClass
    properties
        val
    end
    methods
        function obj = FolderClass(x)
            obj.val = x;
            obj.val = privateHelper(obj);
        end
    end
end
