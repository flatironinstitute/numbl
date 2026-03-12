% Test that calling an inherited static method on a subclass instance
% works through the runtime JIT path (resolveClassMethod).
% TechChild inherits techPref from TechBase (defined in external file).

obj = TechChild();
% techPref is a static method defined in TechBase, inherited by TechChild.
% Calling it on an instance of TechChild should resolve via inheritance.
result = obj.techPref;
assert(isstruct(result), 'Expected struct from inherited static method');
assert(result.alpha == 10, sprintf('Expected alpha=10, got %g', result.alpha));
assert(result.beta == 20, sprintf('Expected beta=20, got %g', result.beta));

disp('SUCCESS');
