% exist and which resolve names in the caller's scope: variables, local,
% nested and private functions, classes, package members and builtins.

% Variables
x = 3;
assert(exist('x') == 1);
assert(strcmp(which('x'), 'variable'));

% Builtins and constants
assert(exist('sin') == 5);
assert(exist('pi') == 5);
assert(~isempty(which('sin')));
assert(~isempty(which('pi')));

% Local function of this script
assert(exist('nr_script_loc') == 2);
assert(~isempty(which('nr_script_loc')));
assert(exist('nr_script_loc', 'file') == 2);
assert(exist('nr_script_loc', 'builtin') == 0);
assert(exist('nr_script_loc', 'var') == 0);

% Private function of this folder
assert(exist('nr_priv') == 2);
assert(~isempty(which('nr_priv')));

% Classes: a classdef file is reported as a file, a package class as a class
assert(exist('NrClass') == 2);
assert(exist('NrClass', 'class') == 8);
assert(exist('NrClass', 'file') == 2);
assert(exist('nrpkg.NrPkgClass') == 8);
assert(exist('nrpkg.NrPkgClass', 'class') == 8);
assert(exist('nrpkg.NrPkgClass', 'file') == 0);
assert(~isempty(which('NrClass')));
assert(~isempty(which('nrpkg.NrPkgClass')));

% Package functions and class members are found by which, not by exist
assert(exist('nrpkg.nr_pkgfn') == 0);
assert(exist('nrpkg.inner.nr_innerfn') == 0);
assert(exist('nrpkg') == 0);
assert(~isempty(which('nrpkg.nr_pkgfn')));
assert(exist('NrClass.twice') == 0);
assert(~isempty(which('NrClass.twice')));
assert(~isempty(which('nrpkg.NrPkgClass.thrice')));
assert(isempty(which('NrColor.Red')));

% Unknown names
assert(exist('nr_no_such_name') == 0);
assert(isempty(which('nr_no_such_name')));
w = which('nr_no_such_name', '-all');
assert(iscell(w) && isempty(w));
w = which('nr_priv', '-all');
assert(iscell(w) && numel(w) == 1);

% Inside a class method: its file's local functions, not its methods
o = NrClass();
assert(o.existLocal() == 2);
assert(o.whichLocalFound());
assert(isequal(o.existMethods(), [0 0]));

% Inside a private function: its locals and its private siblings
r = nr_priv(1);
assert(r.existLocal == 2);
assert(r.existSibling == 2);
assert(r.whichFound);

% Inside a function file: private, local and nested functions
c = nr_caller();
assert(c.existPriv == 2);
assert(c.existPrivFile == 2);
assert(c.whichPrivFound);
assert(c.existLocal == 2);
assert(c.whichLocalFound);
assert(c.existNested == 2);
assert(c.whichNestedFound);

% Under imports
w = nr_wildcard_import();
assert(w.existFn == 0);
assert(w.whichFnFound);
e = nr_explicit_import();
assert(e.existClass == 8);

disp('SUCCESS')

function r = nr_script_loc()
    r = 1;
end
