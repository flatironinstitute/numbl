classdef VarargOutVec
% A simple vector class with a static factory that uses varargout.

    properties
        data
    end

    methods
        function obj = VarargOutVec(d)
            if nargin > 0
                obj.data = d;
            else
                obj.data = [];
            end
        end

        function r = scale(obj, s)
            % Instance method: scale the data
            r = VarargOutVec(obj.data * s);
        end
    end

    methods (Static)
        function varargout = create(varargin)
            % Static factory method using varargout.
            % create(d)      -> returns VarargOutVec(d)
            % [a, b] = create(d1, d2) -> returns two VarargOutVec objects
            for k = 1:nargin
                varargout{k} = VarargOutVec(varargin{k});
            end
        end
    end
end
