/// <reference types="@webgpu/types" />
/**
 * The gpuArray module of the GPU helper (see src/gpuHelper.ts): it owns the
 * WebGPU buffers that hold gpuArray data, keyed by ids the client chooses
 * (so an upload needs no reply). Loaded with `helper.load`, which runs it
 * from its source text, so it must be self-contained.
 */

export function gpuArrayModule() {
  const buffers = new Map<number, GPUBuffer>();
  const usage = () =>
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

  function buffer(id: number): GPUBuffer {
    const b = buffers.get(id);
    if (!b) throw new Error("gpuArray: buffer " + id + " does not exist");
    return b;
  }

  // Freed buffers are kept for reuse by size (up to POOL_BYTES), which
  // saves the allocation and the zero-fill WebGPU gives a new buffer.
  // Every kernel writes all of its output.
  const pool = new Map<number, GPUBuffer[]>();
  let pooled = 0;
  const POOL_BYTES = 512 * 2 ** 20;

  function create(device: GPUDevice, id: number, n: number): GPUBuffer {
    const size = Math.max(4, n * 4);
    const free = pool.get(size);
    let b = free?.pop();
    if (b) pooled -= size;
    else b = device.createBuffer({ size, usage: usage() });
    buffers.set(id, b);
    return b;
  }

  function recycle(b: GPUBuffer): void {
    if (pooled + b.size > POOL_BYTES) {
      b.destroy();
      return;
    }
    pooled += b.size;
    let list = pool.get(b.size);
    if (!list) pool.set(b.size, (list = []));
    list.push(b);
  }

  // ── Kernels ─────────────────────────────────────────────────────────
  // WGSL is generated per operation and operand kinds and compiled once.
  // Values are f32; a logical value is 0 or 1. NaN is tested on the bits,
  // since a shader compiler may assume x == x.

  const PRELUDE = `
fn isnan_(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) > 0x7f800000u; }
fn b2f(c: bool) -> f32 { return select(0.0, 1.0, c); }
fn round_(x: f32) -> f32 {
  let a = abs(x);
  let r = floor(a);
  return sign(x) * select(r, r + 1.0, a - r >= 0.5);
}
fn max_(a: f32, b: f32) -> f32 { return select(select(max(a, b), a, isnan_(b)), b, isnan_(a)); }
fn min_(a: f32, b: f32) -> f32 { return select(select(min(a, b), a, isnan_(b)), b, isnan_(a)); }
fn isinf_(x: f32) -> bool { return (bitcast<u32>(x) & 0x7fffffffu) == 0x7f800000u; }
// MATLAB's mod: the sign of b, mod(a, 0) = a.
fn mod_(a: f32, b: f32) -> f32 {
  if (b == 0.0) { return a; }
  let r = a - floor(a / b) * b;
  return select(r, 0.0, abs(r) >= abs(b));
}
// MATLAB's rem: the sign of a, rem(a, 0) = NaN.
fn rem_(a: f32, b: f32) -> f32 {
  let r = a - trunc(a / b) * b;
  return select(r, 0.0, abs(r) >= abs(b));
}
// sqrt(a^2 + b^2) without overflow; Inf if either is Inf.
fn hypot_(a: f32, b: f32) -> f32 {
  let x = abs(a);
  let y = abs(b);
  if (isinf_(x) || isinf_(y)) { return select(x, y, isinf_(y)); }
  let m = max(x, y);
  if (m == 0.0) { return 0.0; }
  let u = x / m;
  let v = y / m;
  return m * sqrt(u * u + v * v);
}
fn log1p_(x: f32) -> f32 {
  let u = 1.0 + x;
  return select(log(u) * x / (u - 1.0), x, u == 1.0);
}
// a^b for real results (a >= 0, or b an integer); WGSL's pow needs a > 0.
fn pow_(a: f32, b: f32) -> f32 {
  if (b == 0.0) { return 1.0; }
  if (b == 1.0) { return a; }
  if (b == 2.0) { return a * a; }
  if (isnan_(a) || isnan_(b)) { return a + b; }
  if (a == 0.0) { return select(0.0, 1.0 / a, b < 0.0); }
  let r = pow(abs(a), b);
  let odd = fract(b) == 0.0 && fract(b * 0.5) != 0.0;
  return select(r, -r, a < 0.0 && odd);
}
`;

  /** Element-wise operations: a WGSL expression in `a` (and `b`). */
  const ELEMWISE: Record<string, string> = {
    plus: "a + b",
    minus: "a - b",
    times: "a * b",
    rdivide: "a / b",
    ldivide: "b / a",
    lt: "b2f(a < b)",
    gt: "b2f(a > b)",
    le: "b2f(a <= b)",
    ge: "b2f(a >= b)",
    eq: "b2f(a == b)",
    ne: "b2f(a != b)",
    and: "b2f(a != 0.0 && b != 0.0)",
    or: "b2f(a != 0.0 || b != 0.0)",
    max: "max_(a, b)",
    min: "min_(a, b)",
    uminus: "-a",
    not: "b2f(a == 0.0)",
    abs: "abs(a)",
    exp: "exp(a)",
    sin: "sin(a)",
    cos: "cos(a)",
    tan: "tan(a)",
    sinh: "sinh(a)",
    cosh: "cosh(a)",
    tanh: "tanh(a)",
    floor: "floor(a)",
    ceil: "ceil(a)",
    fix: "trunc(a)",
    round: "round_(a)",
    sign: "sign(a)",
    sqrt: "sqrt(a)",
    log: "log(a)",
    log2: "log2(a)",
    log10: "log(a) * 0.4342944819032518",
    log1p: "log1p_(a)",
    asin: "asin(a)",
    acos: "acos(a)",
    atanh: "atanh(a)",
    acosh: "acosh(a)",
    power: "pow_(a, b)",
    isnan: "b2f(isnan_(a))",
    isinf: "b2f(isinf_(a))",
    isfinite: "b2f(!isnan_(a) && !isinf_(a))",
    mod: "mod_(a, b)",
    rem: "rem_(a, b)",
    atan2: "atan2(a, b)",
    hypot: "hypot_(a, b)",
  };

  /** Reductions: `t` maps an element (any/all test it against 0), `step`
   *  folds it into `acc`, `better` (max/min with positions) says whether
   *  x replaces acc (strictly, so the first extreme wins; NaN skipped). */
  const REDUCE: Record<string, { t: string; step: string; better?: string }> = {
    sum: { t: "x", step: "acc + x" },
    prod: { t: "x", step: "acc * x" },
    max: {
      t: "x",
      step: "max_(acc, x)",
      better: "!isnan_(x) && (isnan_(acc) || x > acc)",
    },
    min: {
      t: "x",
      step: "min_(acc, x)",
      better: "!isnan_(x) && (isnan_(acc) || x < acc)",
    },
    any: { t: "b2f(x != 0.0)", step: "max(acc, x)" },
    all: { t: "b2f(x != 0.0)", step: "min(acc, x)" },
  };

  /** For the operations whose real result needs the input in a domain: the
   *  condition under which an element would need a complex result. */
  const DOMAIN: Record<string, string> = {
    sqrt: "a < 0.0",
    log: "a < 0.0",
    log2: "a < 0.0",
    log10: "a < 0.0",
    log1p: "a < -1.0",
    asin: "abs(a) > 1.0",
    acos: "abs(a) > 1.0",
    atanh: "abs(a) > 1.0",
    acosh: "a < 1.0",
    power: "a < 0.0 && fract(b) != 0.0",
  };

  const pipelines = new Map<string, GPUComputePipeline>();
  function pipeline(device: GPUDevice, code: string): GPUComputePipeline {
    let p = pipelines.get(code);
    if (!p) {
      p = device.createComputePipeline({
        layout: "auto",
        compute: {
          module: device.createShaderModule({ code: PRELUDE + code }),
          entryPoint: "main",
        },
      });
      pipelines.set(code, p);
    }
    return p;
  }

  /** Run `code` with a uniform block of u32/f32 words and storage buffers
   *  (binding 0 the uniform, then `bufs` in order), over `groups`
   *  workgroups ([x, y]). */
  async function run(
    device: GPUDevice,
    code: string,
    words: ArrayBuffer,
    bufs: GPUBuffer[],
    groups: [number, number]
  ) {
    // WebGPU reports errors asynchronously; catch this kernel's.
    device.pushErrorScope("validation");
    device.pushErrorScope("out-of-memory");
    const p = pipeline(device, code);
    const uni = device.createBuffer({
      size: Math.max(16, words.byteLength),
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(uni, 0, words);
    const entries: GPUBindGroupEntry[] = [
      { binding: 0, resource: { buffer: uni } },
    ];
    bufs.forEach((b, i) =>
      entries.push({ binding: i + 1, resource: { buffer: b } })
    );
    const bg = device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries,
    });
    const enc = device.createCommandEncoder();
    const pass = enc.beginComputePass();
    pass.setPipeline(p);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(groups[0], groups[1]);
    pass.end();
    device.queue.submit([enc.finish()]);
    uni.destroy();
    const oom = await device.popErrorScope();
    const invalid = await device.popErrorScope();
    const err = oom ?? invalid;
    if (err) throw new Error("gpuArray kernel: " + err.message);
  }

  /** Workgroups of 256 covering `n` threads, and the thread count of one
   *  row of the grid (index = x + y * width). */
  function grid(n: number): { groups: [number, number]; width: number } {
    const g = Math.max(1, Math.ceil(n / 256));
    const x = Math.min(g, 65535);
    return { groups: [x, Math.ceil(g / x)], width: x * 256 };
  }

  /** Uniform words: u32 then f32 values, padded to 16 bytes. */
  function uniforms(u: number[], f: number[] = []): ArrayBuffer {
    const words = Math.ceil((u.length + f.length) / 4) * 4 || 4;
    const buf = new ArrayBuffer(words * 4);
    new Uint32Array(buf, 0, u.length).set(u);
    new Float32Array(buf, u.length * 4, f.length).set(f);
    return buf;
  }

  type Operand =
    | { id: number; n: number; strides?: number[] }
    | { scalar: number };

  /** A storage buffer holding `data`. */
  function words32(
    device: GPUDevice,
    data: Uint32Array<ArrayBuffer>
  ): GPUBuffer {
    const b = device.createBuffer({
      size: Math.max(4, data.byteLength),
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(b, 0, data);
    return b;
  }

  /** The first `n` u32 words of `src` (which allows COPY_SRC). */
  async function readWords(
    device: GPUDevice,
    src: GPUBuffer,
    n: number
  ): Promise<Uint32Array> {
    const staging = device.createBuffer({
      size: n * 4,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    const enc = device.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, staging, 0, n * 4);
    device.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const out = new Uint32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  return {
    info(device: GPUDevice) {
      const a = (
        device as GPUDevice & {
          adapterInfo?: {
            vendor?: string;
            architecture?: string;
            description?: string;
          };
        }
      ).adapterInfo;
      const parts = a ? [a.vendor, a.architecture].filter(Boolean) : [];
      return {
        name: parts.length ? "WebGPU (" + parts.join(" ") + ")" : "WebGPU",
        maxBufferSize: device.limits.maxStorageBufferBindingSize,
      };
    },

    upload(
      device: GPUDevice,
      args: { id: number; data: Float32Array<ArrayBuffer> }
    ) {
      const b = create(device, args.id, args.data.length);
      if (args.data.byteLength > 0) device.queue.writeBuffer(b, 0, args.data);
    },

    async download(device: GPUDevice, args: { id: number; n: number }) {
      const bytes = args.n * 4;
      if (bytes === 0) return new Float32Array(0);
      const src = buffer(args.id);
      const staging = device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(src, 0, staging, 0, bytes);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const out = new Float32Array(staging.getMappedRange().slice(0));
      staging.unmap();
      staging.destroy();
      return out;
    },

    free(_device: GPUDevice, args: { ids: number[] }) {
      for (const id of args.ids) {
        const b = buffers.get(id);
        if (b) recycle(b);
        buffers.delete(id);
      }
    },

    /** out = op(inputs) element-wise over `n` elements. An input is a
     *  buffer of `n` elements, a buffer of 1 (broadcast), or a scalar. */
    async elementwise(
      device: GPUDevice,
      args: {
        out: number;
        op: string;
        n: number;
        inputs: Operand[];
        dims?: number[];
        check?: boolean;
      }
    ) {
      const expr = ELEMWISE[args.op];
      if (!expr) throw new Error("gpuArray: no kernel for " + args.op);
      const out = create(device, args.out, args.n);
      if (args.n === 0) return { bad: false };
      // With `dims` (at most 4), the operands are broadcast: each buffer
      // operand has strides over the result's dimensions (0 where it has
      // size 1).
      const bcast = !!args.dims;
      const decls: string[] = [];
      const reads: string[] = [];
      const bufs: GPUBuffer[] = [out];
      const scalars: number[] = [];
      const strides: number[][] = [];
      args.inputs.forEach((x, k) => {
        const v = "ab"[k];
        if ("scalar" in x) {
          reads.push(`let ${v} = p.s${scalars.length};`);
          scalars.push(x.scalar);
          strides.push([0, 0, 0, 0]);
        } else {
          const binding = bufs.length + 1;
          bufs.push(buffer(x.id));
          decls.push(
            `@group(0) @binding(${binding}) var<storage, read> in${k}: array<f32>;`
          );
          const idx = bcast ? `o${k}` : x.n === 1 ? "0u" : "i";
          reads.push(`let ${v} = in${k}[${idx}];`);
          strides.push(x.strides ?? [0, 0, 0, 0]);
        }
      });
      while (strides.length < 2) strides.push([0, 0, 0, 0]);
      const check = args.check ? DOMAIN[args.op] : undefined;
      let flag: GPUBuffer | null = null;
      if (check) {
        flag = device.createBuffer({
          size: 4,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
        });
        decls.push(
          `@group(0) @binding(${bufs.length + 1}) var<storage, read_write> flag: atomic<u32>;`
        );
        bufs.push(flag);
      }
      const index = bcast
        ? `var r = i; var o0 = 0u; var o1 = 0u;
  for (var d = 0u; d < 4u; d = d + 1u) {
    let sub = r % p.dims[d]; r = r / p.dims[d];
    o0 = o0 + sub * p.sa[d]; o1 = o1 + sub * p.sb[d];
  }`
        : "";
      const { groups, width } = grid(args.n);
      const code = `
struct P { n: u32, w: u32, s0: f32, s1: f32, dims: vec4u, sa: vec4u, sb: vec4u }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
${decls.join("\n")}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + g.y * p.w;
  if (i >= p.n) { return; }
  ${index}
  ${reads.join(" ")}
  ${check ? `if (${check}) { atomicStore(&flag, 1u); }` : ""}
  out[i] = ${expr};
}`;
      while (scalars.length < 2) scalars.push(0);
      const dims = args.dims ?? [1, 1, 1, 1];
      const words = new ArrayBuffer(64);
      new Uint32Array(words, 0, 2).set([args.n, width]);
      new Float32Array(words, 8, 2).set(scalars);
      new Uint32Array(words, 16, 12).set([
        ...dims,
        ...strides[0],
        ...strides[1],
      ]);
      await run(device, code, words, bufs, groups);
      if (!flag) return { bad: false };
      const bad = (await readWords(device, flag, 1))[0] !== 0;
      flag.destroy();
      return { bad };
    },

    /** out[k] = src[off(k)] for the selection's offsets: `start + k*step`,
     *  or `offsets[k]`. */
    async take(
      device: GPUDevice,
      args: {
        out: number;
        src: number;
        n: number;
        start: number;
        step: number;
        offsets: Uint32Array<ArrayBuffer> | null;
      }
    ) {
      const out = create(device, args.out, args.n);
      if (args.n === 0) return;
      const offs = args.offsets && words32(device, args.offsets);
      const code = `
struct P { n: u32, w: u32, start: u32, step: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
${offs ? "@group(0) @binding(3) var<storage, read> offs: array<u32>;" : ""}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x + g.y * p.w;
  if (k >= p.n) { return; }
  dst[k] = src[${offs ? "offs[k]" : "p.start + k * p.step"}];
}`;
      const { groups, width } = grid(args.n);
      const bufs = [out, buffer(args.src)];
      if (offs) bufs.push(offs);
      await run(
        device,
        code,
        uniforms([args.n, width, args.start, args.step]),
        bufs,
        groups
      );
      offs?.destroy();
    },

    /** out = src (`total` elements) with out[off(k)] = rhs for the
     *  selection's `n` offsets; rhs is a buffer of `n` elements, a buffer
     *  of 1, or a scalar. */
    async put(
      device: GPUDevice,
      args: {
        out: number;
        src: number;
        total: number;
        n: number;
        start: number;
        step: number;
        offsets: Uint32Array<ArrayBuffer> | null;
        rhs: Operand;
      }
    ) {
      const out = create(device, args.out, args.total);
      if (args.total === 0) return;
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(buffer(args.src), 0, out, 0, args.total * 4);
      device.queue.submit([enc.finish()]);
      if (args.n === 0) return;
      const offs = args.offsets && words32(device, args.offsets);
      const rhs = args.rhs;
      const scalar = "scalar" in rhs;
      const code = `
struct P { n: u32, w: u32, start: u32, step: u32, s: f32, p1: f32, p2: f32, p3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
${scalar ? "" : "@group(0) @binding(2) var<storage, read> rhs: array<f32>;"}
${offs ? `@group(0) @binding(${scalar ? 2 : 3}) var<storage, read> offs: array<u32>;` : ""}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x + g.y * p.w;
  if (k >= p.n) { return; }
  dst[${offs ? "offs[k]" : "p.start + k * p.step"}] = ${scalar ? "p.s" : !scalar && rhs.n === 1 ? "rhs[0]" : "rhs[k]"};
}`;
      const { groups, width } = grid(args.n);
      const bufs = [out];
      if (!scalar) bufs.push(buffer(rhs.id));
      if (offs) bufs.push(offs);
      await run(
        device,
        code,
        uniforms(
          [args.n, width, args.start, args.step],
          [scalar ? rhs.scalar : 0, 0, 0, 0]
        ),
        bufs,
        groups
      );
      offs?.destroy();
    },

    /** out = op over the middle dimension of `src` viewed as
     *  [inner, len, outer] (column-major): sum, prod, max or min (NaN
     *  skipped), any or all (logical). With `outIdx`, max and min also
     *  give the (0-based) position of the first extreme value. A long
     *  reduction with few outputs runs in several passes. */
    async reduce(
      device: GPUDevice,
      args: {
        out: number;
        outIdx?: number;
        op: string;
        src: number;
        inner: number;
        len: number;
        outer: number;
      }
    ) {
      const nOut = args.inner * args.outer;
      const out = create(device, args.out, nOut);
      const outIdx =
        args.outIdx !== undefined ? create(device, args.outIdx, nOut) : null;
      if (nOut === 0) return;
      const r = REDUCE[args.op];
      if (!r) throw new Error("gpuArray: no reduction " + args.op);
      const withIdx = outIdx !== null;
      // `first`: the pass over the data (positions are k); later passes
      // combine partial results (positions read from the partials).
      const code = (first: boolean) => `
struct P { inner: u32, len: u32, outer: u32, chunk: u32, nchunks: u32, w: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
${withIdx ? "@group(0) @binding(3) var<storage, read_write> dstIdx: array<f32>;" : ""}
${withIdx && !first ? "@group(0) @binding(4) var<storage, read> srcIdx: array<f32>;" : ""}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let t = g.x + g.y * p.w;
  let nOut = p.inner * p.outer;
  if (t >= nOut * p.nchunks) { return; }
  let o = t % nOut;
  let c = t / nOut;
  let ii = o % p.inner;
  let oo = o / p.inner;
  let k0 = c * p.chunk;
  let k1 = min(k0 + p.chunk, p.len);
  let base = ii + oo * p.inner * p.len;
  var x = src[base + k0 * p.inner];
  var acc: f32 = ${r.t};
  ${withIdx ? `var ai: f32 = ${first ? "f32(k0)" : "srcIdx[base + k0 * p.inner]"};` : ""}
  for (var k = k0 + 1u; k < k1; k = k + 1u) {
    x = src[base + k * p.inner];
    ${
      withIdx
        ? `if (${r.better}) { acc = x; ai = ${first ? "f32(k)" : "srcIdx[base + k * p.inner]"}; }`
        : `x = ${r.t}; acc = ${r.step};`
    }
  }
  dst[o + c * nOut] = acc;
  ${withIdx ? "dstIdx[o + c * nOut] = ai;" : ""}
}`;
      let src = buffer(args.src);
      let srcIdx: GPUBuffer | null = null;
      let len = args.len;
      let geom = { inner: args.inner, outer: args.outer };
      let first = true;
      const temps: GPUBuffer[] = [];
      // Split the reduction when there are too few outputs to fill the GPU.
      while (nOut < 4096 && len > 64) {
        const chunk = 64;
        const nchunks = Math.ceil(len / chunk);
        const part = device.createBuffer({
          size: nOut * nchunks * 4,
          usage: usage(),
        });
        const partIdx = withIdx
          ? device.createBuffer({ size: nOut * nchunks * 4, usage: usage() })
          : null;
        temps.push(part);
        if (partIdx) temps.push(partIdx);
        const { groups, width } = grid(nOut * nchunks);
        const bufs = [part, src];
        if (partIdx) bufs.push(partIdx);
        if (partIdx && !first) bufs.push(srcIdx!);
        await run(
          device,
          code(first),
          uniforms([geom.inner, len, geom.outer, chunk, nchunks, width]),
          bufs,
          groups
        );
        // The partials, [nOut, nchunks], reduce over their second
        // dimension: inner = nOut, len = nchunks, outer = 1.
        src = part;
        srcIdx = partIdx;
        len = nchunks;
        geom = { inner: nOut, outer: 1 };
        first = false;
      }
      const { groups, width } = grid(nOut);
      const bufs = [out, src];
      if (outIdx) bufs.push(outIdx);
      if (outIdx && !first) bufs.push(srcIdx!);
      await run(
        device,
        code(first),
        uniforms([geom.inner, len, geom.outer, len, 1, width]),
        bufs,
        groups
      );
      for (const b of temps) b.destroy();
    },

    /** Concatenation: a new `out` of `total` elements, into which each
     *  part (`block` contiguous elements per slab, `outer` slabs) is copied
     *  at `offset` within each slab of `stride` elements. */
    async concat(
      device: GPUDevice,
      args: {
        out: number;
        total: number;
        outer: number;
        stride: number;
        parts: { src: number; block: number; offset: number }[];
      }
    ) {
      const out = create(device, args.out, args.total);
      const code = `
struct P { n: u32, w: u32, block: u32, stride: u32, offset: u32, p1: u32, p2: u32, p3: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let t = g.x + g.y * p.w;
  if (t >= p.n) { return; }
  let o = t / p.block;
  let j = t % p.block;
  dst[o * p.stride + p.offset + j] = src[t];
}`;
      for (const part of args.parts) {
        const n = part.block * args.outer;
        if (n === 0) continue;
        const { groups, width } = grid(n);
        await run(
          device,
          code,
          uniforms([n, width, part.block, args.stride, part.offset, 0, 0, 0]),
          [out, buffer(part.src)],
          groups
        );
      }
    },

    /** Cumulative sum or product over the middle dimension of `src`
     *  viewed as [inner, len, outer]: each sequence is scanned in chunks,
     *  the chunk totals are scanned, and each chunk adds (or multiplies by)
     *  the total before it. */
    async scan(
      device: GPUDevice,
      args: {
        out: number;
        op: string;
        src: number;
        inner: number;
        len: number;
        outer: number;
      }
    ) {
      const total = args.inner * args.len * args.outer;
      const out = create(device, args.out, total);
      if (total === 0) return;
      const f = args.op === "cumprod" ? "*" : "+";
      const unit = args.op === "cumprod" ? "1.0" : "0.0";
      const nseq = args.inner * args.outer;
      const chunk = 256;
      const nchunks = Math.ceil(args.len / chunk);
      const head = `
struct P { inner: u32, len: u32, outer: u32, chunk: u32, nchunks: u32, w: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
fn at(s: u32, k: u32) -> u32 {
  return (s % p.inner) + k * p.inner + (s / p.inner) * p.inner * p.len;
}`;
      // 1: scan within each chunk; record the chunk's total.
      const local = `${head}
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> totals: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let t = g.x + g.y * p.w;
  if (t >= p.inner * p.outer * p.nchunks) { return; }
  let s = t % (p.inner * p.outer);
  let c = t / (p.inner * p.outer);
  let k1 = min((c + 1u) * p.chunk, p.len);
  var acc: f32 = ${unit};
  for (var k = c * p.chunk; k < k1; k = k + 1u) {
    acc = acc ${f} src[at(s, k)];
    dst[at(s, k)] = acc;
  }
  totals[s + c * p.inner * p.outer] = acc;
}`;
      // 2: per sequence, the running total before each chunk (in place).
      const across = `${head}
@group(0) @binding(1) var<storage, read_write> totals: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let s = g.x + g.y * p.w;
  let nseq = p.inner * p.outer;
  if (s >= nseq) { return; }
  var acc: f32 = ${unit};
  for (var c = 0u; c < p.nchunks; c = c + 1u) {
    let v = totals[s + c * nseq];
    totals[s + c * nseq] = acc;
    acc = acc ${f} v;
  }
}`;
      // 3: combine each element with its chunk's preceding total.
      const fix = `${head}
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> totals: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let t = g.x + g.y * p.w;
  let nseq = p.inner * p.outer;
  if (t >= nseq * p.len) { return; }
  let s = t % nseq;
  let k = t / nseq;
  let c = k / p.chunk;
  if (c == 0u) { return; }
  let i = at(s, k);
  dst[i] = totals[s + c * nseq] ${f} dst[i];
}`;
      const totals = device.createBuffer({
        size: nseq * nchunks * 4,
        usage: usage(),
      });
      const u = (n: number) =>
        uniforms([
          args.inner,
          args.len,
          args.outer,
          chunk,
          nchunks,
          grid(n).width,
          0,
          0,
        ]);
      await run(
        device,
        local,
        u(nseq * nchunks),
        [out, buffer(args.src), totals],
        grid(nseq * nchunks).groups
      );
      if (nchunks > 1) {
        await run(device, across, u(nseq), [totals], grid(nseq).groups);
        await run(
          device,
          fix,
          u(nseq * args.len),
          [out, totals],
          grid(nseq * args.len).groups
        );
      }
      totals.destroy();
    },

    /** out (m x n) = a (m x k) * b (k x n), column-major, tiled. */
    async matmul(
      device: GPUDevice,
      args: {
        out: number;
        a: number;
        b: number;
        m: number;
        k: number;
        n: number;
      }
    ) {
      const out = create(device, args.out, args.m * args.n);
      if (args.m * args.n === 0) return;
      const code = `
struct P { m: u32, k: u32, n: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> C: array<f32>;
@group(0) @binding(2) var<storage, read> A: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
var<workgroup> As: array<f32, 256>;
var<workgroup> Bs: array<f32, 256>;
@compute @workgroup_size(16, 16) fn main(
  @builtin(local_invocation_id) l: vec3u,
  @builtin(workgroup_id) w: vec3u
) {
  let row = w.x * 16u + l.x;
  let col = w.y * 16u + l.y;
  var acc: f32 = 0.0;
  let tiles = (p.k + 15u) / 16u;
  for (var t = 0u; t < tiles; t = t + 1u) {
    let ak = t * 16u + l.y;
    var av: f32 = 0.0;
    if (row < p.m && ak < p.k) { av = A[row + ak * p.m]; }
    As[l.y * 16u + l.x] = av;
    let bk = t * 16u + l.x;
    var bv: f32 = 0.0;
    if (bk < p.k && col < p.n) { bv = B[bk + col * p.k]; }
    Bs[l.y * 16u + l.x] = bv;
    workgroupBarrier();
    for (var q = 0u; q < 16u; q = q + 1u) {
      acc = acc + As[q * 16u + l.x] * Bs[l.y * 16u + q];
    }
    workgroupBarrier();
  }
  if (row < p.m && col < p.n) { C[row + col * p.m] = acc; }
}`;
      if (args.k === 0) {
        // An empty inner dimension gives zeros.
        const enc = device.createCommandEncoder();
        enc.clearBuffer(out);
        device.queue.submit([enc.finish()]);
        return;
      }
      await run(
        device,
        code,
        uniforms([args.m, args.k, args.n, 0]),
        [out, buffer(args.a), buffer(args.b)],
        [Math.ceil(args.m / 16), Math.ceil(args.n / 16)]
      );
    },

    /** out (n x m) = transpose of src (m x n). */
    async transpose(
      device: GPUDevice,
      args: { out: number; src: number; m: number; n: number }
    ) {
      const total = args.m * args.n;
      const out = create(device, args.out, total);
      if (total === 0) return;
      const code = `
struct P { m: u32, n: u32, w: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let t = g.x + g.y * p.w;
  if (t >= p.m * p.n) { return; }
  let i = t % p.m;
  let j = t / p.m;
  dst[j + i * p.n] = src[t];
}`;
      const { groups, width } = grid(total);
      await run(
        device,
        code,
        uniforms([args.m, args.n, width, 0]),
        [out, buffer(args.src)],
        groups
      );
    },

    async synchronize(device: GPUDevice) {
      await device.queue.onSubmittedWorkDone();
    },

    dispose() {
      for (const b of buffers.values()) b.destroy();
      buffers.clear();
      for (const list of pool.values()) for (const b of list) b.destroy();
      pool.clear();
      pooled = 0;
    },
  };
}
