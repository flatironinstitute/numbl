% uihtml: render an HTML string as a figure. Runs headless here (the
% instruction is emitted but not rendered); this exercises the shim classdef
% and the drawuihtml native primitive.

h = uihtml('HTMLSource', '<p style="color:red">Hello <b>numbl</b></p>');
assert(strcmp(h.HTMLSource, '<p style="color:red">Hello <b>numbl</b></p>'));

% leading parent argument is accepted and ignored
h2 = uihtml(1, 'HTMLSource', '<h1>two</h1>');
assert(strcmp(h2.HTMLSource, '<h1>two</h1>'));

% re-render after changing the source
h2.HTMLSource = '<h1>updated</h1>';
h2.show();

disp('SUCCESS')
