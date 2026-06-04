classdef uihtml < handle
    %UIHTML Create an HTML UI component (numbl subset).
    %   H = UIHTML('HTMLSource', HTML) renders the self-contained HTML string
    %   HTML in the figure pane. An optional leading parent argument (e.g. a
    %   figure) is accepted and ignored. Supported name-value options:
    %   HTMLSource, Data, Position.
    %
    %   H = UIHTML('HTMLSource', HTML, 'Data', X) also sends X into the page.
    %   Mirroring MATLAB, X is encoded with jsonencode, parsed in the page with
    %   JSON.parse, and set on the JavaScript `htmlComponent.Data` object,
    %   firing any "DataChanged" listener registered in the page's
    %   `function setup(htmlComponent)`. To update the data after construction,
    %   set H.Data and call show(H) (numbl re-renders the component):
    %
    %       h = uihtml('HTMLSource', html, 'Data', struct('n', 1));
    %       h.Data = struct('n', 2);
    %       show(h);
    %
    %   numbl currently supports HTMLSource given as an HTML markup string (a
    %   single self-contained document). HTML file paths and supporting files
    %   are not yet supported.
    %
    %   The reverse channel (page -> MATLAB) is supported for the IDE:
    %   HTMLEventReceivedFcn fires when JS calls
    %   htmlComponent.sendEventToMATLAB(name,data); inside the callback use
    %   sendEventToHTMLSource(src,name,data) to send back to the page.
    %   DataChangedFcn fires when JS sets htmlComponent.Data. Register callbacks
    %   at construction (name-value) since numbl renders at construction.
    properties
        HTMLSource = ''
        Data = []
        Position = [100 100 100 100]
        HTMLEventReceivedFcn = []
        DataChangedFcn = []
    end
    methods
        function obj = uihtml(varargin)
            args = varargin;
            % Ignore an optional leading parent argument (anything that is not
            % a name-value name string).
            if numel(args) >= 1 && ~(ischar(args{1}) || isstring(args{1}))
                args = args(2:end);
            end
            for i = 1:2:numel(args) - 1
                name = args{i};
                val = args{i + 1};
                if strcmpi(name, 'HTMLSource')
                    obj.HTMLSource = val;
                elseif strcmpi(name, 'Data')
                    obj.Data = val;
                elseif strcmpi(name, 'Position')
                    obj.Position = val;
                elseif strcmpi(name, 'HTMLEventReceivedFcn')
                    obj.HTMLEventReceivedFcn = val;
                elseif strcmpi(name, 'DataChangedFcn')
                    obj.DataChangedFcn = val;
                end
            end
            render(obj);
        end

        function show(obj)
            render(obj);
        end

        function render(obj)
            if isempty(obj.HTMLSource)
                return;
            end
            if isempty(obj.Data)
                id = drawuihtml(obj.HTMLSource);
            else
                id = drawuihtml(obj.HTMLSource, jsonencode(obj.Data));
            end
            if ~isempty(obj.HTMLEventReceivedFcn)
                registeruihtmlcallback(id, 'HTMLEventReceived', ...
                                       obj.HTMLEventReceivedFcn);
            end
            if ~isempty(obj.DataChangedFcn)
                registeruihtmlcallback(id, 'DataChanged', obj.DataChangedFcn);
            end
        end
    end
end
