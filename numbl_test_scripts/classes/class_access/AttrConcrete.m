classdef AttrConcrete < AttrAbstract
    % Fixture for class_access.m: implements every abstract member.
    properties
        ap = 3
    end
    methods
        function r = doIt(obj)
            r = obj.ap * 2;
        end
    end
end
