% Dotted names: package functions and classes, nested packages, static
% methods, enumeration members and Class.empty, directly, through imports
% and through function handles.

assert(nrpkg.nr_pkgfn(1) == 8);
assert(nrpkg.inner.nr_innerfn(1) == 71);
assert(nrpkg.NrPkgClass.thrice(2) == 6);
assert(NrClass.twice(4) == 8);
assert(NrColor.Green == NrColor.pick(2));
assert(isequal(size(NrClass.empty(0, 3)), [0 3]));

% Handles to static methods, package functions and nested packages
h = @NrClass.twice;
assert(h(4) == 8);
h = @nrpkg.NrPkgClass.thrice;
assert(h(4) == 12);
h = @nrpkg.inner.nr_innerfn;
assert(h(1) == 71);
h = @nrpkg.nr_pkgfn;
assert(h(1) == 8);
h = @NrColor.pick;
assert(h(3) == NrColor.Blue);
h = @NrClass.empty;
assert(isequal(size(h(0, 2)), [0 2]));
h = @() NrColor.Green;
assert(h() == NrColor.Green);
assert(isequal(cellfun(@NrClass.twice, {1, 2}), [2 4]));

% A handle to an enumeration member is created but cannot be called
h = @NrColor.Red;
assert(isa(h, 'function_handle'));
failed = false;
try
    h();
catch
    failed = true;
end
assert(failed);

% feval with a dotted name
assert(feval('nrpkg.nr_pkgfn', 2) == 9);
assert(feval('NrClass.twice', 5) == 10);

% Imports
w = nr_wildcard_import();
assert(w.static == 6);
assert(w.staticHandle == 15);
assert(w.fn == 8);
assert(w.fnHandle == 9);
assert(w.subpackage == 71);
assert(w.feval == 11);
assert(isequal(w.emptySize, [0 1]));

e = nr_explicit_import();
assert(e.static == 6);
assert(e.staticHandle == 15);
assert(strcmp(e.className, 'nrpkg.NrPkgClass'));

disp('SUCCESS')
