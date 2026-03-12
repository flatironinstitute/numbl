classdef PrefStore3_
% Test class that mimics the chebfunpref constructor pattern:
% - Constructor accepts optional class instance via varargin
% - outObj = varargin{1} widens the type from ClassInstance to Unknown
% - Then outObj.prefList = val must set the declared property directly,
%   NOT route through subsasgn (even though type is Unknown).

    properties
        prefList
    end

    methods
        function obj = PrefStore3_(varargin)
            if nargin > 0 && isa(varargin{1}, 'PrefStore3_')
                % Copy constructor — this widens the type to Unknown
                obj = varargin{1};
            end
            % Direct assignment to declared property prefList.
            % Even though obj's type may be Unknown (after varargin{1}),
            % this should NOT go through subsasgn.
            obj.prefList = struct();
            obj.prefList.alpha = 10;
            obj.prefList.techPrefs = struct();
            obj.prefList.techPrefs.x = 1;
            obj.prefList.techPrefs.y = 2;
        end

        function out = subsref(obj, S)
            switch S(1).type
                case '.'
                    if strcmp(S(1).subs, 'prefList')
                        out = obj.prefList;
                        if numel(S) > 1
                            out = builtin('subsref', out, S(2:end));
                        end
                    elseif isfield(obj.prefList, S(1).subs)
                        out = obj.prefList.(S(1).subs);
                        if numel(S) > 1
                            out = builtin('subsref', out, S(2:end));
                        end
                    else
                        error('No field %s', S(1).subs);
                    end
                otherwise
                    error('Unsupported subscript type');
            end
        end

        function obj = subsasgn(obj, S, val)
            switch S(1).type
                case '.'
                    if isfield(obj.prefList, S(1).subs)
                        obj.prefList = builtin('subsasgn', obj.prefList, S, val);
                    else
                        obj.prefList.techPrefs = builtin('subsasgn', ...
                            obj.prefList.techPrefs, S, val);
                    end
                otherwise
                    error('Unsupported subscript type');
            end
        end
    end
end
