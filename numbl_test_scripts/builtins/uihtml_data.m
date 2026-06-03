% uihtml Data bridge: construct with a 'Data' name-value, and update it via
% h.Data + show(h). Runs headless (the instruction is emitted but not
% rendered); this exercises the shim's jsonencode of Data and the drawuihtml
% second argument. The DataChanged firing itself is browser-side.

html = ['<div id="out">x</div>' ...
        '<script>function setup(c){' ...
        'c.addEventListener("DataChanged",function(e){' ...
        'document.getElementById("out").innerHTML=e.Data.label;});}' ...
        '</script>'];

% Data given at construction (a struct -> JSON object in the page).
d = struct('label', 'hello', 'value', 42);
h = uihtml('HTMLSource', html, 'Data', d);
assert(strcmp(h.Data.label, 'hello'));
assert(h.Data.value == 42);

% Update the data, then re-render with show().
h.Data = struct('label', 'world', 'value', 7);
assert(strcmp(h.Data.label, 'world'));
h.show();

% A component with no Data still constructs (setup runs, no DataChanged).
h2 = uihtml('HTMLSource', '<p>no data</p>');
assert(isempty(h2.Data));

disp('SUCCESS')
