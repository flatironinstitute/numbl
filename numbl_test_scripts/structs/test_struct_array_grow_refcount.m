% Regression: growing a struct array element and assigning its 3rd+ field
% used to underflow the refcount of a shared default-empty tensor sentinel
% (the missing-field backfill set fields without an incref).

% Inline growth across iterations with 3+ fields, including empty defaults.
plan = struct();
plan(1).type = 'a'; plan(1).name = 'x'; plan(1).default = []; plan(1).v = 1;
plan(2).type = 'b'; plan(2).name = 'y'; plan(2).default = []; plan(2).v = 2;
plan(3).type = 'c'; plan(3).name = 'z'; plan(3).default = 5; plan(3).v = 3;
assert(numel(plan) == 3, 'struct array length');
assert(strcmp(plan(2).name, 'y'), 'field readback name');
assert(plan(3).v == 3, 'field readback v');
assert(isempty(plan(1).default), 'empty default preserved');

% Growth through nested function returns (the matlab2tikz m2tInputParser shape).
ipp.plan = {};
ipp = addopt(ipp, 'filename', '', @(x) true);
ipp = addopt(ipp, 'filehandle', [], @(x) false);
assert(numel(ipp.plan) == 2, 'nested-grow length');
assert(strcmp(ipp.plan(2).name, 'filehandle'), 'nested-grow field');

disp('SUCCESS');

function p = addopt(q, name, default, validator)
  p = q;
  plan = p.plan;
  if isempty(plan)
    plan = struct();
    n = 1;
  else
    n = numel(plan) + 1;
  end
  plan(n).type = 'optional';
  plan(n).name = name;
  plan(n).default = default;
  plan(n).validator = validator;
  p.plan = plan;
end
