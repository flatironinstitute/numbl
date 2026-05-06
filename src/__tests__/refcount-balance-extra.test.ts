import { describe, it, expect } from "vitest";
import { executeCode } from "../numbl-core/executeCode.js";

/**
 * Extended balance probe — broader builtin coverage than the core
 * `refcount-balance.test.ts`. Each script ends in `clear;` and the
 * pool is asserted to be fully balanced afterward.
 *
 * When adding a new builtin, add a one-line script here so the next
 * regression in its memory-pool bookkeeping shows up in CI.
 */

const cases: Array<[string, string]> = [
  // ── linear algebra / decompositions ────────────────────────────────────
  ["inv", "x = inv([1 2; 3 5]); clear;"],
  ["det", "x = det([1 2; 3 4]); clear;"],
  ["norm", "x = norm([3 4]); clear;"],
  ["trace", "x = trace([1 2; 3 4]); clear;"],
  ["eig", "x = eig([1 2; 2 1]); clear;"],
  ["svd", "x = svd([1 2; 3 4]); clear;"],
  ["chol", "x = chol([4 2; 2 3]); clear;"],
  ["qr", "[Q,R] = qr([1 2; 3 4]); clear;"],
  ["lu", "[L,U] = lu([1 2; 3 4]); clear;"],
  ["mldivide", "x = [1 2;3 4]\\[5;6]; clear;"],
  ["mrdivide", "x = [5 6]/[1 2;3 4]; clear;"],
  ["pinv", "x = pinv([1 2; 3 4; 5 6]); clear;"],
  ["rank", "x = rank([1 2; 2 4]); clear;"],

  // ── arithmetic / element-wise ──────────────────────────────────────────
  ["pow elem", "x = [1 2 3].^2; clear;"],
  ["pow matrix", "x = [1 2; 3 4]^2; clear;"],
  ["unary minus", "x = -[1 2 3]; clear;"],
  ["right divide", "x = [4 6 8] ./ 2; clear;"],
  ["mod", "x = mod([5 7 9], 3); clear;"],
  ["rem", "x = rem([5 7 9], 3); clear;"],
  ["floor", "x = floor([1.4 2.6 3.5]); clear;"],
  ["ceil", "x = ceil([1.4 2.6 3.5]); clear;"],
  ["round", "x = round([1.4 2.6 3.5]); clear;"],
  ["sign", "x = sign([-1 0 1]); clear;"],
  ["sqrt", "x = sqrt([4 9 16]); clear;"],
  ["exp", "x = exp([0 1 2]); clear;"],
  ["log", "x = log([1 2 3]); clear;"],
  ["log2", "x = log2([1 2 4 8]); clear;"],
  ["log10", "x = log10([1 10 100]); clear;"],

  // ── trig / hyp ─────────────────────────────────────────────────────────
  ["cos", "x = cos([0 1 2]); clear;"],
  ["tan", "x = tan([0 1 2]); clear;"],
  ["asin", "x = asin([0 0.5 1]); clear;"],
  ["acos", "x = acos([0 0.5 1]); clear;"],
  ["atan", "x = atan([0 1 2]); clear;"],
  ["atan2", "x = atan2([1 2], [3 4]); clear;"],
  ["sinh", "x = sinh([0 1 2]); clear;"],
  ["cosh", "x = cosh([0 1 2]); clear;"],

  // ── reductions ────────────────────────────────────────────────────────
  ["prod", "x = prod([1 2 3 4]); clear;"],
  ["std", "x = std([1 2 3 4 5]); clear;"],
  ["var", "x = var([1 2 3 4 5]); clear;"],
  ["median", "x = median([3 1 4 1 5]); clear;"],
  ["any", "x = any([0 0 1]); clear;"],
  ["all", "x = all([1 1 1]); clear;"],
  ["cumsum", "x = cumsum([1 2 3 4]); clear;"],
  ["cumprod", "x = cumprod([1 2 3 4]); clear;"],
  ["diff", "x = diff([1 3 6 10]); clear;"],
  ["sum all", "x = sum([1 2 3 4], 'all'); clear;"],
  ["max with dim", "x = max([1 2; 3 4], [], 1); clear;"],
  ["max two-arg", "x = max([1 2 3], [3 2 1]); clear;"],
  ["min two-arg", "x = min([1 2 3], [3 2 1]); clear;"],
  ["max [val,idx]", "[v, i] = max([3 1 4 1 5]); clear;"],

  // ── indexing / slicing ─────────────────────────────────────────────────
  ["colon", "a = [1 2 3 4]; b = a(:); clear;"],
  ["linear index", "a = [1 2; 3 4]; b = a(3); clear;"],
  ["logical index", "a = [1 2 3 4]; b = a([true false true false]); clear;"],
  ["end keyword", "a = [1 2 3 4]; b = a(end-1); clear;"],
  ["2d index", "a = [1 2 3; 4 5 6]; b = a(2, :); clear;"],
  ["negative index", "a = [1 2 3]; a(end+1) = 4; clear;"],

  // ── reshape / shape ops ────────────────────────────────────────────────
  ["transpose complex", "x = [1+2i, 3+4i]'; clear;"],
  ["transpose real", "x = [1 2 3]'; clear;"],
  ["nonconj transpose", "x = [1+2i, 3+4i].'; clear;"],
  ["fliplr", "x = fliplr([1 2 3 4]); clear;"],
  ["flipud", "x = flipud([1; 2; 3; 4]); clear;"],
  ["rot90", "x = rot90([1 2; 3 4]); clear;"],
  ["circshift", "x = circshift([1 2 3 4], 1); clear;"],
  ["permute", "x = permute([1 2; 3 4], [2 1]); clear;"],
  ["kron", "x = kron([1 2], [3 4]); clear;"],
  ["cat", "x = cat(1, [1 2], [3 4]); clear;"],
  ["horzcat", "x = horzcat([1 2], [3 4]); clear;"],
  ["vertcat", "x = vertcat([1 2], [3 4]); clear;"],

  // ── creation ──────────────────────────────────────────────────────────
  ["zeros", "x = zeros(3, 4); clear;"],
  ["ones", "x = ones(3, 4); clear;"],
  ["eye", "x = eye(4); clear;"],
  ["rand", "x = rand(3, 4); clear;"],
  ["randn", "x = randn(3, 4); clear;"],
  ["range vector", "x = 1:10; clear;"],
  ["linspace big", "x = linspace(0, 1, 200); clear;"],
  ["logspace", "x = logspace(0, 2, 5); clear;"],
  ["true(n)", "x = true(3); clear;"],
  ["false(n)", "x = false(3); clear;"],

  // ── set operations ─────────────────────────────────────────────────────
  ["union", "x = union([1 2 3], [2 3 4]); clear;"],
  ["intersect", "x = intersect([1 2 3], [2 3 4]); clear;"],
  ["setdiff", "x = setdiff([1 2 3], [2 3 4]); clear;"],
  ["ismember", "x = ismember([1 2 3], [2 3 4]); clear;"],

  // ── classes (handle / value) ───────────────────────────────────────────
  [
    "class instance",
    `
    classdef Foo
      properties
        x
      end
      methods
        function obj = Foo(v); obj.x = v; end
      end
    end
    f = Foo([1 2 3]);
    clear;
    `,
  ],
  [
    "handle class",
    `
    classdef Bar < handle
      properties
        x
      end
      methods
        function obj = Bar(v); obj.x = v; end
      end
    end
    b = Bar([1 2 3]); b.x = [9 9];
    clear;
    `,
  ],

  // ── error handling ────────────────────────────────────────────────────
  [
    "nested try-catch",
    "try; try; x=[1 2 3]; error('inner'); catch; y=[4 5]; error('outer'); end; catch; z=[9 9]; end; clear;",
  ],

  // ── strings/chars ─────────────────────────────────────────────────────
  ["sprintf with tensor", "x = sprintf('%d ', [1 2 3 4]); clear;"],
  ["num2str", "x = num2str([1.5 2.5]); clear;"],
  ["str2num", "x = str2num('1 2 3'); clear;"],

  // ── globals/persistents ────────────────────────────────────────────────
  // (skipped: `clear g` doesn't remove from rt.$g per MATLAB semantics, so
  // the global keeps its tensor live for the runtime's lifetime.)

  // ── operator coverage ──────────────────────────────────────────────────
  ["bitand", "x = bitand(uint8([1 2 3]), uint8(2)); clear;"],
  ["complex extraction", "z = [1+2i, 3+4i]; r = real(z); i_ = imag(z); clear;"],
  ["abs complex", "x = abs([1+2i, 3+4i]); clear;"],
  ["conj", "x = conj([1+2i, 3+4i]); clear;"],
  ["angle", "x = angle([1+2i, 3+4i]); clear;"],

  // ── conversions ────────────────────────────────────────────────────────
  ["int32 from tensor", "x = int32([1.5 2.5 3.5]); clear;"],
  ["uint8 from tensor", "x = uint8([1 2 3]); clear;"],
  ["logical from tensor", "x = logical([0 1 2]); clear;"],
  ["isempty", "x = isempty([1 2 3]); clear;"],
  ["isnumeric", "x = isnumeric([1 2 3]); clear;"],
  ["isnan", "x = isnan([1 NaN 3]); clear;"],
  ["isinf", "x = isinf([1 Inf 3]); clear;"],
  ["isfinite", "x = isfinite([1 Inf 3]); clear;"],

  // ── misc ──────────────────────────────────────────────────────────────
  ["sortrows", "x = sortrows([3 1; 1 2; 2 3]); clear;"],
  // histcounts not implemented yet
  ["accumarray", "x = accumarray([1; 2; 1; 3], [10; 20; 30; 40]); clear;"],
  ["find with cond", "x = find([1 2 3 4] > 2); clear;"],
  ["find limit", "x = find([1 2 3 4 5], 2); clear;"],

  // ── sparse ────────────────────────────────────────────────────────────
  ["sparse from triplet", "S = sparse([1 2], [1 2], [3 4]); clear;"],
  ["full from sparse", "S = sparse([1 2], [1 2], [3 4]); F = full(S); clear;"],
  ["sparse mul", "S = sparse([1 2], [1 2], [3 4]); x = S * [1; 1]; clear;"],

  // ── sort variants ─────────────────────────────────────────────────────
  ["sort desc", "x = sort([3 1 4 1 5], 'descend'); clear;"],
  ["sort with idx", "[v, i] = sort([3 1 4 1 5]); clear;"],

  // ── FFT family ────────────────────────────────────────────────────────
  ["fft", "x = fft([1 2 3 4]); clear;"],
  ["ifft", "x = ifft(fft([1 2 3 4])); clear;"],
  // fft2 not implemented yet
  ["fftshift", "x = fftshift(fft([1 2 3 4 5 6 7 8])); clear;"],

  // ── conv ───────────────────────────────────────────────────────────────
  ["conv", "x = conv([1 2 3], [1 1]); clear;"],
  // conv2 not implemented yet

  // ── tri ops ───────────────────────────────────────────────────────────
  ["tril", "x = tril([1 2 3; 4 5 6; 7 8 9]); clear;"],
  ["triu", "x = triu([1 2 3; 4 5 6; 7 8 9]); clear;"],
  ["diag from vec", "x = diag([1 2 3]); clear;"],
  ["diag from mat", "x = diag([1 2 3; 4 5 6; 7 8 9]); clear;"],
  ["trace", "x = trace([1 2; 3 4]); clear;"],

  // ── matrix exp ─────────────────────────────────────────────────────────
  // expm/sqrtm not implemented yet

  // ── stats ─────────────────────────────────────────────────────────────
  ["cov", "x = cov([1 2; 3 4; 5 6]); clear;"],
  ["corrcoef", "x = corrcoef([1 2; 3 4; 5 6]); clear;"],

  // ── NaN ops ────────────────────────────────────────────────────────────
  ["nansum", "x = sum([1 NaN 3], 'omitnan'); clear;"],
  ["nanmean", "x = mean([1 NaN 3], 'omitnan'); clear;"],

  // ── deep chains ───────────────────────────────────────────────────────
  ["deep chain", "x = sum(sin(cos([1 2 3 4 5]))) + max(abs([6 7 8])); clear;"],
  [
    "nested anon call",
    "f = @(x) sum(x.^2); g = @(x) f(x) + sum(x); y = g([1 2 3 4]); clear;",
  ],
  ["matrix chain", "A = [1 2;3 4]; x = A * A' * A * inv(A); clear;"],

  // ── random sampling ────────────────────────────────────────────────────
  ["randperm", "x = randperm(10); clear;"],
  ["randi", "x = randi(100, 1, 10); clear;"],

  // ── slicing / indexing more ────────────────────────────────────────────
  ["3d tensor index", "a = ones(2, 3, 4); b = a(:, :, 2); clear;"],
  ["nested index", "a = {[1 2 3], [4 5 6]}; b = a{1}(2); clear;"],
  ["deep struct", "s.a.b.c = [1 2 3]; clear;"],
  [
    "array of cells",
    "c = cell(1, 3); c{1} = [1 2]; c{2} = [3 4]; c{3} = [5 6]; clear;",
  ],

  // ── error in middle ───────────────────────────────────────────────────
  [
    "error then continue",
    "try; a = [1 2 3]; error('e'); catch; end; b = [4 5 6]; clear;",
  ],

  // ── mixed types ───────────────────────────────────────────────────────
  ["mixed bool tensor", "x = ([1 2 3] > 1) + [4 5 6]; clear;"],
  ["complex divide", "x = (1+2i) ./ [1 2 3]; clear;"],

  // ── multi-step programs ────────────────────────────────────────────────
  [
    "3-step program",
    "a = zeros(50,1); for k=1:50; a(k) = k^2; end; b = sum(a); clear;",
  ],
  ["polynomial roots", "x = roots([1 -3 2]); clear;"],
  ["polynomial value", "x = polyval([1 2 3], [1 2 3]); clear;"],

  // ── colon tricky cases ─────────────────────────────────────────────────
  ["A(:) reshape", "A = reshape(1:24, 2, 3, 4); B = A(:); clear;"],
  ["logical slicing", "a = 1:10; b = a(a > 5); clear;"],
  ["scalar struct array", "s(1).x = 1; s(2).x = 2; t = s(1).x; clear;"],
];

describe("refcount balance: extended builtins", () => {
  for (const [name, src] of cases) {
    it(name, () => {
      const r = executeCode(src, { optimization: "0" });
      const s = r.memoryStats!;
      expect(s.liveSetSize, `liveSetSize should be 0; src=${src}`).toBe(0);
      expect(s.attemptedAllocs).toBe(s.releases);
    });
  }
});
