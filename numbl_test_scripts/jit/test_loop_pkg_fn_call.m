% Package-namespace function call inside a JIT'd loop.
%
% Mirrors chunkie's adapgausskerneval pattern:
%   nint = chnk.perp(dint);
% where `chnk` is a +chnk/ package directory, NOT a local variable, and
% `chnk.perp` is a workspace function defined in +chnk/perp.m. The
% parser produces a `MethodCall` AST node for this; the JIT must
% recognize the package-prefix shape and route through the regular
% user-function call path.
%
% The %!numbl:assert_jit directive in the loop body asserts the loop
% IS JIT-compiled. Without MethodCall support, the lowering bails on
% "MethodCall: unsupported expression" and the loop runs interpreted.

% --- 1) Bare package call returning a tensor.
%
% perp's body uses `[tau(2,:); -tau(1,:)]` + reshape — the reshape may
% prevent the callee from inlining, but soft-bail to UserDispatchCall
% should keep the outer loop JIT'd as long as the probe succeeds.
tau_base = [1 2 3 4 5; 10 20 30 40 50];  % 2x5, no Range expressions
total = 0;
for ii = 1:5
    %!numbl:assert_jit
    n = pkgcall_helpers.perp(tau_base);
    total = total + n(1, 1) + ii;
end
% perp returns [tau(2,:); -tau(1,:)] so n(1, 1) == tau_base(2, 1) == 10.
% Loop adds 10 each iter plus ii=1..5.
expected = 10 * 5 + (1 + 2 + 3 + 4 + 5);
assert(total == expected, '1: perp call inside JIT loop, scalar accumulator');

% --- 2) Package call where the callee returns a scalar (tightens
%        soft-bail probe path).
total2 = 0;
x = ones(2, 4);
for ii = 1:5
    %!numbl:assert_jit
    total2 = total2 + pkgcall_helpers.scale(x, ii);
end
assert(total2 == 8 * (1 + 2 + 3 + 4 + 5), '2: package scalar fn');

% --- 3) Strict-dispatch check: when the package prefix IS shadowed by a
%        local variable, the JIT must NOT route to a workspace function.
%        Here `pkgcall_helpers` is bound to a struct with a field-handle
%        named `perp`; `pkgcall_helpers.perp(tau)` becomes a struct field
%        deref + call, NOT a package call. We don't expect this to JIT
%        with our MethodCall fix (it's the wrong dispatch). Verifying it
%        produces the same RUNTIME answer is enough.
pkgcall_helpers = struct('perp', @(t) [-t(2,:); t(1,:)]);
got = pkgcall_helpers.perp([1; 2]);
assert(isequal(got, [-2; 1]), '3: shadowed prefix dispatches to struct field handle');
clear pkgcall_helpers;

disp('SUCCESS');
