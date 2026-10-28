classdef MetaForms_
    % Fixture for class_metaclass.m: other attribute syntaxes.
    properties (Access = ?MetaChild_)
        a = 1
    end
    properties (~Hidden, SetAccess = 'private')
        b = 2
    end
    methods (Access = {?MetaChild_, ?MetaBase_})
        function m(obj)
        end
    end
end
