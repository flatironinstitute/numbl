classdef inputParser < handle
    properties
        CaseSensitive = false
        FunctionName = ''
        KeepUnmatched = false
        PartialMatching = true
        StructExpand = true
        Results = struct()
        Unmatched = struct()
        UsingDefaults = {}
        Parameters = {}
    end
    properties
        required_ = {}
        optional_ = {}
        params_ = {}
    end
    methods
        function addRequired(obj, name, validator)
            if nargin < 3
                validator = [];
            end
            entry = struct('name', name, 'default', {[]}, 'validator', {{}});
            if ~isempty(validator)
                entry.validator = validator;
            end
            obj.required_{end+1} = entry;
            obj.Parameters{end+1} = name;
        end

        function addOptional(obj, name, default, validator)
            if nargin < 4
                validator = [];
            end
            entry = struct('name', name, 'default', {default}, 'validator', {{}});
            if ~isempty(validator)
                entry.validator = validator;
            end
            obj.optional_{end+1} = entry;
            obj.Parameters{end+1} = name;
        end

        function addParameter(obj, name, default, validator)
            if nargin < 4
                validator = [];
            end
            entry = struct('name', name, 'default', {default}, 'validator', {{}});
            if ~isempty(validator)
                entry.validator = validator;
            end
            obj.params_{end+1} = entry;
            obj.Parameters{end+1} = name;
        end

        function addParamValue(obj, name, default, validator)
            addParameter(obj, name, default, validator);
        end

        function parse(obj, varargin)
            s = struct();
            usingDefaults = {};
            pos = 1;
            nargs = numel(varargin);

            % Parse required arguments
            for i = 1:numel(obj.required_)
                entry = obj.required_{i};
                if pos > nargs
                    error('Not enough input arguments. Missing required argument ''%s''.', entry.name);
                end
                val = varargin{pos};
                pos = pos + 1;
                s.(entry.name) = val;
            end

            % Parse optional positional arguments
            % An optional arg is consumed only if it is NOT a string matching a parameter name
            for i = 1:numel(obj.optional_)
                entry = obj.optional_{i};
                if pos <= nargs
                    nextVal = varargin{pos};
                    isParamName = false;
                    if ischar(nextVal) || isstring(nextVal)
                        paramName = char(nextVal);
                        for k = 1:numel(obj.params_)
                            pentry = obj.params_{k};
                            if inputParser.namesMatch(paramName, pentry.name, obj.CaseSensitive)
                                isParamName = true;
                                break;
                            end
                        end
                    end
                    if ~isParamName
                        s.(entry.name) = nextVal;
                        pos = pos + 1;
                    else
                        s.(entry.name) = entry.default;
                        usingDefaults{end+1} = entry.name;
                    end
                else
                    s.(entry.name) = entry.default;
                    usingDefaults{end+1} = entry.name;
                end
            end

            % Set defaults for all parameters
            for i = 1:numel(obj.params_)
                entry = obj.params_{i};
                s.(entry.name) = entry.default;
                usingDefaults{end+1} = entry.name;
            end

            % Parse name-value pairs from remaining arguments
            unmatched = struct();
            while pos <= nargs
                if pos + 1 > nargs
                    error('Expected name-value pair argument.');
                end
                name = varargin{pos};
                val = varargin{pos + 1};
                pos = pos + 2;

                if ~ischar(name) && ~isstring(name)
                    error('Expected parameter name to be a string.');
                end
                name = char(name);

                found = false;
                for i = 1:numel(obj.params_)
                    entry = obj.params_{i};
                    if inputParser.namesMatch(name, entry.name, obj.CaseSensitive)
                        s.(entry.name) = val;
                        % Remove from usingDefaults
                        newDefaults = {};
                        for j = 1:numel(usingDefaults)
                            if ~strcmp(usingDefaults{j}, entry.name)
                                newDefaults{end+1} = usingDefaults{j};
                            end
                        end
                        usingDefaults = newDefaults;
                        found = true;
                        break;
                    end
                end
                if ~found
                    if obj.KeepUnmatched
                        unmatched.(name) = val;
                    else
                        error('''%s'' is not a recognized parameter.', name);
                    end
                end
            end

            obj.Results = s;
            obj.UsingDefaults = usingDefaults;
            obj.Unmatched = unmatched;
        end
    end

    methods (Static)
        function tf = namesMatch(input, expected, caseSensitive)
            if caseSensitive
                tf = strcmp(input, expected);
            else
                tf = strcmpi(input, expected);
            end
        end
    end
end
