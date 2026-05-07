% appdata storage must manage refcounts on stored containers.
%
% Regression: setappdata used to keep a JS reference to the value
% without increfing it. Once the caller's local went out of scope the
% container hit rc=0 and _destroy decref'd its children. A later
% getappdata then bound the (already destroyed) wrapper, and the
% caller's clearLocals walked into the dead children — producing
% "refcount underflow on char (rc=0)" under strictRefcount.

% ── 1. Store a cell of chars, drop callers, fetch and use ───────────
function set_packages()
  pkgs = {'gh/owner/foo', 'gh/owner/bar', 'gh/owner/baz'};
  setappdata(0, 'PKGS', pkgs);
end

function out = get_packages()
  out = getappdata(0, 'PKGS');
end

set_packages();
% At this point the only ref to the cell is the appdata bucket. If
% setappdata didn't incref, the cell would already be destroyed.
got = get_packages();
assert(iscell(got), '1a: got cell back');
assert(numel(got) == 3, '1b: three elements');
assert(strcmp(got{1}, 'gh/owner/foo'), '1c: first element preserved');
assert(strcmp(got{3}, 'gh/owner/baz'), '1d: third element preserved');

% ── 2. ismember + getappdata round-trip mirrors mip.is_loaded ───────
function tf = is_member(name)
  bucket = getappdata(0, 'PKGS');
  tf = ismember(name, bucket);
end

assert(is_member('gh/owner/bar'), '2a: present member');
assert(~is_member('gh/owner/missing'), '2b: absent member');

% ── 3. Replacing an existing key drops the old value cleanly ────────
function replace_packages()
  setappdata(0, 'PKGS', {'new/a', 'new/b'});
end

replace_packages();
got2 = getappdata(0, 'PKGS');
assert(numel(got2) == 2, '3a: replaced cell length');
assert(strcmp(got2{1}, 'new/a'), '3b: replaced first element');

% ── 4. rmappdata releases the value, isappdata reflects removal ─────
rmappdata(0, 'PKGS');
assert(~isappdata(0, 'PKGS'), '4a: key removed');

% ── 5. Stress: many set/get cycles with cells inside cells ──────────
function set_nested(n)
  inner = cell(1, n);
  for k = 1:n
    inner{k} = sprintf('item-%d', k);
  end
  setappdata(0, 'NESTED', {inner, 'tail'});
end

function v = first_inner()
  outer = getappdata(0, 'NESTED');
  v = outer{1}{1};
end

for trial = 1:20
  set_nested(5);
  assert(strcmp(first_inner(), 'item-1'), sprintf('5: trial %d', trial));
end
rmappdata(0, 'NESTED');

disp('SUCCESS');
