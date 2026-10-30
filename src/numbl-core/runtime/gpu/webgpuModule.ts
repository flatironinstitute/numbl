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
      discard(b);
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
// Complex values are vec2f (real, imaginary).
fn cmul(a: vec2f, b: vec2f) -> vec2f { return vec2f(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
// Smith's algorithm; a real divisor divides each part.
fn cdiv(a: vec2f, b: vec2f) -> vec2f {
  if (b.y == 0.0) { return a / b.x; }
  if (abs(b.x) >= abs(b.y)) {
    let r = b.y / b.x;
    let d = b.x + b.y * r;
    return vec2f((a.x + a.y * r) / d, (a.y - a.x * r) / d);
  }
  let r = b.x / b.y;
  let d = b.x * r + b.y;
  return vec2f((a.x * r + a.y) / d, (a.y * r - a.x) / d);
}
fn cabs(a: vec2f) -> f32 { return hypot_(a.x, a.y); }
fn cangle(a: vec2f) -> f32 { return select(atan2(a.y, a.x), 0.0, a.x == 0.0 && a.y == 0.0); }
fn cexp(a: vec2f) -> vec2f {
  let e = exp(a.x);
  if (a.y == 0.0) { return vec2f(e, 0.0); }
  return vec2f(e * cos(a.y), e * sin(a.y));
}
fn clog(a: vec2f) -> vec2f { return vec2f(log(cabs(a)), cangle(a)); }
// The principal square root.
fn csqrt(a: vec2f) -> vec2f {
  let r = cabs(a);
  if (r == 0.0) { return vec2f(0.0, a.y); }
  let t = sqrt((r + abs(a.x)) * 0.5);
  if (a.x >= 0.0) { return vec2f(t, a.y / (2.0 * t)); }
  return vec2f(abs(a.y) / (2.0 * t), select(t, -t, a.y < 0.0));
}
fn cpow(a: vec2f, b: vec2f) -> vec2f {
  if (b.y == 0.0) {
    if (b.x == 0.0) { return vec2f(1.0, 0.0); }
    if (b.x == 1.0) { return a; }
    if (b.x == 2.0) { return cmul(a, a); }
    if (a.y == 0.0 && (a.x >= 0.0 || fract(b.x) == 0.0)) { return vec2f(pow_(a.x, b.x), 0.0); }
  }
  if (a.x == 0.0 && a.y == 0.0) {
    return select(vec2f(abs(1.0 / a.x), 0.0), vec2f(0.0, 0.0), b.x > 0.0);
  }
  return cexp(cmul(b, clog(a)));
}
fn csin(a: vec2f) -> vec2f { return vec2f(sin(a.x) * cosh(a.y), cos(a.x) * sinh(a.y)); }
fn ccos(a: vec2f) -> vec2f { return vec2f(cos(a.x) * cosh(a.y), -sin(a.x) * sinh(a.y)); }
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

  /** Complex element-wise operations: a WGSL expression in the vec2f
   *  values `a` (and `b`), a vec2f or, for an operation with a real
   *  result, an f32. `ra` and `rb` are the forms for a real `a` or `b`,
   *  which treat an infinite part as the host does. */
  const CELEMWISE: Record<string, { e: string; ra?: string; rb?: string }> = {
    plus: { e: "a + b" },
    minus: { e: "a - b" },
    times: { e: "cmul(a, b)", ra: "a.x * b", rb: "a * b.x" },
    rdivide: { e: "cdiv(a, b)" },
    ldivide: { e: "cdiv(b, a)" },
    // conj(a) .* b, for dot.
    conjtimes: { e: "cmul(vec2f(a.x, -a.y), b)", ra: "a.x * b" },
    uminus: { e: "-a" },
    conj: { e: "vec2f(a.x, -a.y)" },
    complex: { e: "vec2f(a.x, b.x)" },
    exp: { e: "cexp(a)" },
    log: { e: "clog(a)" },
    sqrt: { e: "csqrt(a)" },
    power: { e: "cpow(a, b)" },
    sin: { e: "csin(a)" },
    cos: { e: "ccos(a)" },
    real: { e: "a.x" },
    imag: { e: "a.y" },
    abs: { e: "cabs(a)" },
    angle: { e: "cangle(a)" },
    eq: { e: "b2f(a.x == b.x && a.y == b.y)" },
    ne: { e: "b2f(!(a.x == b.x && a.y == b.y))" },
    isnan: { e: "b2f(isnan_(a.x) || isnan_(a.y))" },
    isinf: { e: "b2f(isinf_(a.x) || isinf_(a.y))" },
    isfinite: {
      e: "b2f(!isnan_(a.x) && !isnan_(a.y) && !isinf_(a.x) && !isinf_(a.y))",
    },
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

  // ── Batches ─────────────────────────────────────────────────────────
  // Kernels are encoded into one command buffer, submitted when a request
  // needs results (`settle`), after BATCH dispatches, or once the posted
  // calls already received have run (a timer). Their uniforms share one
  // buffer, written at submit (queue order makes reusing it safe). A batch
  // runs in one pair of error scopes, popped without waiting; the next
  // request raises an error. A buffer that queued work may read is
  // destroyed after the submit (`discard`), and an upload into a reused
  // buffer submits first.

  const BATCH = 128;
  const UNI_STRIDE = 256;
  let enc: GPUCommandEncoder | null = null;
  let pass: GPUComputePassEncoder | null = null;
  let dispatches = 0;
  const uniData = new Uint8Array(BATCH * UNI_STRIDE);
  let uniBuf: GPUBuffer | null = null;
  const doomed: GPUBuffer[] = [];
  let scopes: Promise<(GPUError | Error | null)[]>[] = [];

  /** The batch's command encoder, outside a compute pass. */
  function encoder(device: GPUDevice): GPUCommandEncoder {
    if (!enc) {
      device.pushErrorScope("validation");
      device.pushErrorScope("out-of-memory");
      enc = device.createCommandEncoder();
      setTimeout(() => {
        try {
          flush(device);
        } catch {
          // The device is gone (shutdown).
        }
      }, 0);
    }
    if (pass) {
      pass.end();
      pass = null;
    }
    return enc;
  }

  function flush(device: GPUDevice): void {
    if (!enc) return;
    if (pass) pass.end();
    pass = null;
    if (dispatches > 0)
      device.queue.writeBuffer(uniBuf!, 0, uniData, 0, dispatches * UNI_STRIDE);
    device.queue.submit([enc.finish()]);
    enc = null;
    dispatches = 0;
    scopes.push(
      Promise.all([device.popErrorScope(), device.popErrorScope()]).catch(
        (e: Error) => [e]
      )
    );
    for (const b of doomed) b.destroy();
    doomed.length = 0;
  }

  /** Submit the batch and raise an error from the work submitted. */
  async function settle(device: GPUDevice): Promise<void> {
    flush(device);
    const pending = scopes;
    scopes = [];
    for (const errs of await Promise.all(pending)) {
      const err = errs[0] ?? errs[1];
      if (err) throw new Error("gpuArray kernel: " + err.message);
    }
  }

  /** Destroy `b` once the work that may read it is submitted. */
  function discard(b: GPUBuffer): void {
    if (enc) doomed.push(b);
    else b.destroy();
  }

  /** Encode `code` with a uniform block of u32/f32 words and storage
   *  buffers (binding 0 the uniform, then `bufs` in order), over `groups`
   *  workgroups ([x, y]). */
  function run(
    device: GPUDevice,
    code: string,
    words: ArrayBuffer,
    bufs: GPUBuffer[],
    groups: [number, number]
  ): void {
    if (dispatches === BATCH) flush(device);
    const p = pipeline(device, code);
    const e = enc ?? encoder(device);
    if (!pass) pass = e.beginComputePass();
    uniBuf ??= device.createBuffer({
      size: uniData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const offset = dispatches * UNI_STRIDE;
    uniData.set(new Uint8Array(words), offset);
    const entries: GPUBindGroupEntry[] = [
      {
        binding: 0,
        resource: {
          buffer: uniBuf,
          offset,
          size: Math.max(16, words.byteLength),
        },
      },
    ];
    bufs.forEach((b, i) =>
      entries.push({ binding: i + 1, resource: { buffer: b } })
    );
    const bg = device.createBindGroup({
      layout: p.getBindGroupLayout(0),
      entries,
    });
    pass.setPipeline(p);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(groups[0], groups[1]);
    dispatches++;
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

  /** A selection of `take` and `put` (see `KernelSelection`). */
  type Sel = {
    n: number;
    start: number;
    step: number;
    offsets: Uint32Array<ArrayBuffer> | null;
    grid?: { counts: number[]; strides: number[] };
  };

  /** WGSL for a selection's k-th offset, given the uniform fields `start`
   *  and `step`, or (`grid`) `gc` and `gs`, and the buffer `offs`. */
  function selOffset(sel: Sel): string {
    if (sel.offsets) return "offs[k]";
    if (!sel.grid) return "p.start + k * p.step";
    return "at(k)";
  }

  const GRID_AT = `
fn at(k: u32) -> u32 {
  var r = k;
  var o = p.start;
  for (var d = 0u; d < 4u; d = d + 1u) {
    o = o + (r % p.gc[d]) * p.gs[d];
    r = r / p.gc[d];
  }
  return o;
}`;

  /** A grid's counts and strides padded to 4 dimensions (counts of 1). */
  function gridWords(sel: Sel): number[] {
    const c = [1, 1, 1, 1];
    const s = [0, 0, 0, 0];
    sel.grid?.counts.forEach((v, d) => (c[d] = v));
    sel.grid?.strides.forEach((v, d) => (s[d] = v));
    return [...c, ...s];
  }

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
    await settle(device);
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

  // ── FFT tables ──────────────────────────────────────────────────────
  // Twiddle factors and Bluestein's chirps are computed in double
  // precision here, stored as vec2f, and kept for reuse.

  const tables = new Map<string, GPUBuffer>();

  function table(
    device: GPUDevice,
    key: string,
    n: number,
    fill: (re: Float64Array, im: Float64Array) => void
  ): GPUBuffer {
    let b = tables.get(key);
    if (!b) {
      const re = new Float64Array(n);
      const im = new Float64Array(n);
      fill(re, im);
      const data = new Float32Array(2 * Math.max(1, n));
      for (let k = 0; k < n; k++) {
        data[2 * k] = re[k];
        data[2 * k + 1] = im[k];
      }
      b = device.createBuffer({
        size: data.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      device.queue.writeBuffer(b, 0, data);
      tables.set(key, b);
    }
    return b;
  }

  /** exp(sign * 2 pi i t / m) for t < m/2. */
  function twiddles(device: GPUDevice, m: number, sign: number): GPUBuffer {
    return table(device, `tw ${m} ${sign}`, m >> 1, (re, im) => {
      for (let t = 0; t < re.length; t++) {
        re[t] = Math.cos((2 * Math.PI * t) / m);
        im[t] = sign * Math.sin((2 * Math.PI * t) / m);
      }
    });
  }

  /** Bluestein's chirp c_j = exp(sign * pi i j^2 / n) for j < n. */
  function chirp(device: GPUDevice, n: number, sign: number): GPUBuffer {
    return table(device, `chirp ${n} ${sign}`, n, (re, im) =>
      chirpValues(n, sign, re, im)
    );
  }

  function chirpValues(
    n: number,
    sign: number,
    re: Float64Array,
    im: Float64Array
  ): void {
    for (let j = 0; j < n; j++) {
      // j^2 mod 2n is exact in double for j < 2^26.
      const a = (Math.PI * ((j * j) % (2 * n))) / n;
      re[j] = Math.cos(a);
      im[j] = sign * Math.sin(a);
    }
  }

  /** The forward FFT (length m) of Bluestein's filter conj(c_j), wrapped
   *  around to negative j. */
  function chirpFilter(
    device: GPUDevice,
    n: number,
    m: number,
    sign: number
  ): GPUBuffer {
    return table(device, `filter ${n} ${m} ${sign}`, m, (re, im) => {
      const cr = new Float64Array(n);
      const ci = new Float64Array(n);
      chirpValues(n, sign, cr, ci);
      for (let j = 0; j < n; j++) {
        re[j] = cr[j];
        im[j] = -ci[j];
        if (j > 0) {
          re[m - j] = cr[j];
          im[m - j] = -ci[j];
        }
      }
      fft64(re, im);
    });
  }

  /** In-place forward FFT in double precision (length a power of 2). */
  function fft64(re: Float64Array, im: Float64Array): void {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        [re[i], re[j]] = [re[j], re[i]];
        [im[i], im[j]] = [im[j], im[i]];
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const a = (-2 * Math.PI) / len;
      for (let i = 0; i < n; i += len)
        for (let k = 0; k < len / 2; k++) {
          const wr = Math.cos(a * k);
          const wi = Math.sin(a * k);
          const p = i + k;
          const q = p + len / 2;
          const xr = re[q] * wr - im[q] * wi;
          const xi = re[q] * wi + im[q] * wr;
          re[q] = re[p] - xr;
          im[q] = im[p] - xi;
          re[p] += xr;
          im[p] += xi;
        }
    }
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
      // Queued work may still read a reused buffer.
      flush(device);
      const b = create(device, args.id, args.data.length);
      if (args.data.byteLength > 0) device.queue.writeBuffer(b, 0, args.data);
    },

    async download(device: GPUDevice, args: { id: number; n: number }) {
      await settle(device);
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
      run(device, code, words, bufs, groups);
      if (!flag) return { bad: false };
      const bad = (await readWords(device, flag, 1))[0] !== 0;
      flag.destroy();
      return { bad };
    },

    /** A fused element-wise expression (see runtime/gpu/fusion.ts): each
     *  node applies an `ELEMWISE` template to inputs, scalars or earlier
     *  nodes, in one kernel. */
    async fused(
      device: GPUDevice,
      args: {
        out: number;
        n: number;
        inputs: { id: number; n: number }[];
        scalars: number[];
        nodes: {
          op: string;
          args: ({ in: number } | { s: number } | { t: number })[];
        }[];
      }
    ) {
      const out = create(device, args.out, args.n);
      if (args.n === 0) return;
      if (args.scalars.length > 16)
        throw new Error("gpuArray: too many scalars");
      const ref = (r: { in: number } | { s: number } | { t: number }) =>
        "in" in r
          ? "x" + r.in
          : "s" in r
            ? `p.s[${r.s >> 2}][${r.s & 3}]`
            : "t" + r.t;
      const lines = args.nodes.map((node, j) => {
        const tpl = ELEMWISE[node.op];
        if (!tpl) throw new Error("gpuArray: no kernel for " + node.op);
        const a = node.args[0] ? ref(node.args[0]) : "";
        const b = node.args[1] ? ref(node.args[1]) : "";
        return `let t${j} = ${tpl.replace(/\b[ab]\b/g, v => (v === "a" ? a : b))};`;
      });
      const decls = args.inputs.map(
        (_x, k) =>
          `@group(0) @binding(${k + 2}) var<storage, read> in${k}: array<f32>;`
      );
      const reads = args.inputs.map(
        (x, k) => `let x${k} = in${k}[${x.n === 1 ? "0u" : "i"}];`
      );
      const code = `
struct P { n: u32, w: u32, p0: u32, p1: u32, s: array<vec4f, 4> }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
${decls.join("\n")}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + g.y * p.w;
  if (i >= p.n) { return; }
  ${reads.join("\n  ")}
  ${lines.join("\n  ")}
  out[i] = t${args.nodes.length - 1};
}`;
      const { groups, width } = grid(args.n);
      const words = new ArrayBuffer(80);
      new Uint32Array(words, 0, 2).set([args.n, width]);
      new Float32Array(words, 16, args.scalars.length).set(args.scalars);
      run(
        device,
        code,
        words,
        [out, ...args.inputs.map(x => buffer(x.id))],
        groups
      );
    },

    /** Complex element-wise: out (and outIm, for a complex result) =
     *  op(inputs) over `n` elements, broadcast as in `elementwise`. An
     *  input is a buffer's real part and imaginary part (0 for real data),
     *  or a complex scalar. */
    async celementwise(
      device: GPUDevice,
      args: {
        out: number;
        outIm: number;
        op: string;
        n: number;
        inputs: (
          | { re: number; im: number; n: number; strides?: number[] }
          | { scalar: [number, number] }
        )[];
        dims?: number[];
      }
    ) {
      const tpl = CELEMWISE[args.op];
      if (!tpl) throw new Error("gpuArray: no complex kernel for " + args.op);
      const cplx = args.outIm !== 0;
      const out = create(device, args.out, args.n);
      const outIm = cplx ? create(device, args.outIm, args.n) : null;
      if (args.n === 0) return;
      const bcast = !!args.dims;
      const decls: string[] = [];
      const reads: string[] = [];
      const bufs: GPUBuffer[] = outIm ? [out, outIm] : [out];
      const scalars = [0, 0, 0, 0];
      const strides: number[][] = [];
      const real: boolean[] = [];
      args.inputs.forEach((x, k) => {
        const v = "ab"[k];
        if ("scalar" in x) {
          reads.push(`let ${v} = p.s.${k === 0 ? "xy" : "zw"};`);
          scalars[2 * k] = x.scalar[0];
          scalars[2 * k + 1] = x.scalar[1];
          strides.push([0, 0, 0, 0]);
          real.push(x.scalar[1] === 0);
          return;
        }
        const idx = bcast ? `o${k}` : x.n === 1 ? "0u" : "i";
        bufs.push(buffer(x.re));
        decls.push(
          `@group(0) @binding(${bufs.length}) var<storage, read> re${k}: array<f32>;`
        );
        let im = "0.0";
        if (x.im) {
          bufs.push(buffer(x.im));
          decls.push(
            `@group(0) @binding(${bufs.length}) var<storage, read> im${k}: array<f32>;`
          );
          im = `im${k}[${idx}]`;
        }
        reads.push(`let ${v} = vec2f(re${k}[${idx}], ${im});`);
        strides.push(x.strides ?? [0, 0, 0, 0]);
        real.push(!x.im);
      });
      while (strides.length < 2) strides.push([0, 0, 0, 0]);
      const expr = (real[0] && tpl.ra) || (real[1] && tpl.rb) || tpl.e;
      const index = bcast
        ? `var r = i; var o0 = 0u; var o1 = 0u;
  for (var d = 0u; d < 4u; d = d + 1u) {
    let sub = r % p.dims[d]; r = r / p.dims[d];
    o0 = o0 + sub * p.sa[d]; o1 = o1 + sub * p.sb[d];
  }`
        : "";
      const { groups, width } = grid(args.n);
      const code = `
struct P { n: u32, w: u32, p0: u32, p1: u32, s: vec4f, dims: vec4u, sa: vec4u, sb: vec4u }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
${cplx ? "@group(0) @binding(2) var<storage, read_write> outIm: array<f32>;" : ""}
${decls.join("\n")}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + g.y * p.w;
  if (i >= p.n) { return; }
  ${index}
  ${reads.join(" ")}
  ${cplx ? `let z = ${expr}; out[i] = z.x; outIm[i] = z.y;` : `out[i] = ${expr};`}
}`;
      const dims = args.dims ?? [1, 1, 1, 1];
      const words = new ArrayBuffer(80);
      new Uint32Array(words, 0, 2).set([args.n, width]);
      new Float32Array(words, 16, 4).set(scalars);
      new Uint32Array(words, 32, 12).set([
        ...dims,
        ...strides[0],
        ...strides[1],
      ]);
      run(device, code, words, bufs, groups);
    },

    /** The DFT (or inverse) of length `n` over the middle dimension of
     *  `src` (real, or complex with `srcIm`) viewed as [inner, len, outer],
     *  into out (and outIm), [inner, n, outer]. The sequences are copied
     *  (padded or truncated) into a work buffer of vec2f, transformed there
     *  by radix-2 Stockham passes, and copied out. A length that is not a
     *  power of 2 uses Bluestein's algorithm: a convolution with a chirp,
     *  by transforms of a power-of-2 length m >= 2n - 1. With `symmetric`,
     *  the second half of each sequence mirrors the first (conjugated), and
     *  only the real part is written. */
    async fft(
      device: GPUDevice,
      args: {
        out: number;
        outIm: number;
        src: number;
        srcIm: number;
        inner: number;
        len: number;
        outer: number;
        n: number;
        inverse: boolean;
        symmetric: boolean;
      }
    ) {
      const { inner, len, outer, n } = args;
      const nseq = inner * outer;
      const out = create(device, args.out, nseq * n);
      const outIm = args.symmetric
        ? null
        : create(device, args.outIm, nseq * n);
      if (nseq * n === 0) return;
      const sign = args.inverse ? 1 : -1;
      const pow2 = (n & (n - 1)) === 0;
      let m = n;
      if (!pow2) {
        m = 1;
        while (m < 2 * n - 1) m *= 2;
      }
      const work = () =>
        device.createBuffer({ size: nseq * m * 8, usage: usage() });
      let a = work();
      let b = work();
      const head = `
struct P { inner: u32, len: u32, n: u32, m: u32, nseq: u32, w: u32, ns: u32, sym: u32, scale: f32, p1: f32, p2: f32, p3: f32 }
@group(0) @binding(0) var<uniform> p: P;`;
      const u = (threads: number, ns = 0, scale = 1) =>
        uniforms(
          [
            inner,
            len,
            n,
            m,
            nseq,
            grid(threads).width,
            ns,
            args.symmetric ? 1 : 0,
          ],
          [scale, 0, 0, 0]
        );
      // Copy in: element q of a sequence is src[q] (0 past len); with
      // `symmetric`, q > n/2 is conj(src[n - q]). Bluestein's input is
      // multiplied by the chirp.
      const cplxIn = args.srcIm !== 0;
      const pack = `${head}
@group(0) @binding(1) var<storage, read_write> dst: array<vec2f>;
@group(0) @binding(2) var<storage, read> srcRe: array<f32>;
${cplxIn ? "@group(0) @binding(3) var<storage, read> srcIm: array<f32>;" : ""}
${pow2 ? "" : `@group(0) @binding(${cplxIn ? 4 : 3}) var<storage, read> chirp: array<vec2f>;`}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let t = g.x + g.y * p.w;
  if (t >= p.nseq * p.m) { return; }
  let s = t / p.m;
  let j = t % p.m;
  var v = vec2f(0.0, 0.0);
  if (j < p.n) {
    var q = j;
    let mirror = p.sym != 0u && 2u * j > p.n;
    if (mirror) { q = p.n - j; }
    if (q < p.len) {
      let at = (s % p.inner) + q * p.inner + (s / p.inner) * p.inner * p.len;
      v = vec2f(srcRe[at], ${cplxIn ? "srcIm[at]" : "0.0"});
      if (mirror) { v.y = -v.y; }
    }
    ${pow2 ? "" : "v = cmul(v, chirp[j]);"}
  }
  dst[t] = v;
}`;
      const packBufs = [a, buffer(args.src)];
      if (cplxIn) packBufs.push(buffer(args.srcIm));
      if (!pow2) packBufs.push(chirp(device, n, sign));
      run(device, pack, u(nseq * m), packBufs, grid(nseq * m).groups);
      // One radix-2 pass: sub-transforms of length ns combine into 2*ns.
      const half = m >> 1;
      const pass = `${head}
@group(0) @binding(1) var<storage, read_write> dst: array<vec2f>;
@group(0) @binding(2) var<storage, read> src: array<vec2f>;
@group(0) @binding(3) var<storage, read> tw: array<vec2f>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let t = g.x + g.y * p.w;
  let half = p.m / 2u;
  if (t >= p.nseq * half) { return; }
  let base = (t / half) * p.m;
  let j = t % half;
  let k = j % p.ns;
  let x0 = src[base + j];
  let x1 = cmul(src[base + j + half], tw[k * (p.m / (2u * p.ns))]);
  let d = base + (j / p.ns) * 2u * p.ns + k;
  dst[d] = x0 + x1;
  dst[d + p.ns] = x0 - x1;
}`;
      const transform = (dir: number) => {
        const tw = twiddles(device, m, dir);
        for (let ns = 1; ns < m; ns *= 2) {
          run(
            device,
            pass,
            u(nseq * half, ns),
            [b, a, tw],
            grid(nseq * half).groups
          );
          [a, b] = [b, a];
        }
      };
      if (pow2) transform(sign);
      else {
        // Convolve with the chirp filter: forward transform, multiply by
        // the filter's transform, inverse transform (scaled below).
        transform(-1);
        const mul = `${head}
@group(0) @binding(1) var<storage, read_write> x: array<vec2f>;
@group(0) @binding(2) var<storage, read> h: array<vec2f>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let t = g.x + g.y * p.w;
  if (t >= p.nseq * p.m) { return; }
  x[t] = cmul(x[t], h[t % p.m]);
}`;
        run(
          device,
          mul,
          u(nseq * m),
          [a, chirpFilter(device, n, m, sign)],
          grid(nseq * m).groups
        );
        transform(1);
      }
      // Copy out the first n elements of each sequence, scaled (and, for
      // Bluestein, multiplied by the chirp).
      const unpack = `${head}
@group(0) @binding(1) var<storage, read_write> outRe: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<vec2f>;
${outIm ? "@group(0) @binding(3) var<storage, read_write> outIm: array<f32>;" : ""}
${pow2 ? "" : `@group(0) @binding(${outIm ? 4 : 3}) var<storage, read> chirp: array<vec2f>;`}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let t = g.x + g.y * p.w;
  if (t >= p.nseq * p.n) { return; }
  let s = t / p.n;
  let k = t % p.n;
  var v = src[s * p.m + k];
  ${pow2 ? "" : "v = cmul(v, chirp[k]);"}
  v = v * p.scale;
  let at = (s % p.inner) + k * p.inner + (s / p.inner) * p.inner * p.n;
  outRe[at] = v.x;
  ${outIm ? "outIm[at] = v.y;" : ""}
}`;
      const scale = (pow2 ? 1 : 1 / m) * (args.inverse ? 1 / n : 1);
      const outBufs = [out, a];
      if (outIm) outBufs.push(outIm);
      if (!pow2) outBufs.push(chirp(device, n, sign));
      run(
        device,
        unpack,
        u(nseq * n, 0, scale),
        outBufs,
        grid(nseq * n).groups
      );
      discard(a);
      discard(b);
    },

    /** out[k] = src[off(k)] for the selection's offsets: `start + k*step`,
     *  `offsets[k]`, or a grid's. */
    async take(device: GPUDevice, args: Sel & { out: number; src: number }) {
      const out = create(device, args.out, args.n);
      if (args.n === 0) return;
      const offs = args.offsets && words32(device, args.offsets);
      const code = `
struct P { n: u32, w: u32, start: u32, step: u32, gc: vec4u, gs: vec4u }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
${offs ? "@group(0) @binding(3) var<storage, read> offs: array<u32>;" : ""}
${args.grid ? GRID_AT : ""}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x + g.y * p.w;
  if (k >= p.n) { return; }
  dst[k] = src[${selOffset(args)}];
}`;
      const { groups, width } = grid(args.n);
      const bufs = [out, buffer(args.src)];
      if (offs) bufs.push(offs);
      run(
        device,
        code,
        uniforms([args.n, width, args.start, args.step, ...gridWords(args)]),
        bufs,
        groups
      );
      if (offs) discard(offs);
    },

    /** out = src (`total` elements) with out[off(k)] = rhs for the
     *  selection's `n` offsets; rhs is a buffer of `n` elements, a buffer
     *  of 1, or a scalar. */
    async put(
      device: GPUDevice,
      args: Sel & { out: number; src: number; total: number; rhs: Operand }
    ) {
      const out = create(device, args.out, args.total);
      if (args.total === 0) return;
      encoder(device).copyBufferToBuffer(
        buffer(args.src),
        0,
        out,
        0,
        args.total * 4
      );
      if (args.n === 0) return;
      const offs = args.offsets && words32(device, args.offsets);
      const rhs = args.rhs;
      const scalar = "scalar" in rhs;
      const code = `
struct P { n: u32, w: u32, start: u32, step: u32, gc: vec4u, gs: vec4u, s: f32, p1: f32, p2: f32, p3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
${scalar ? "" : "@group(0) @binding(2) var<storage, read> rhs: array<f32>;"}
${offs ? `@group(0) @binding(${scalar ? 2 : 3}) var<storage, read> offs: array<u32>;` : ""}
${args.grid ? GRID_AT : ""}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let k = g.x + g.y * p.w;
  if (k >= p.n) { return; }
  dst[${selOffset(args)}] = ${scalar ? "p.s" : !scalar && rhs.n === 1 ? "rhs[0]" : "rhs[k]"};
}`;
      const { groups, width } = grid(args.n);
      const bufs = [out];
      if (!scalar) bufs.push(buffer(rhs.id));
      if (offs) bufs.push(offs);
      run(
        device,
        code,
        uniforms(
          [args.n, width, args.start, args.step, ...gridWords(args)],
          [scalar ? rhs.scalar : 0, 0, 0, 0]
        ),
        bufs,
        groups
      );
      if (offs) discard(offs);
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
        run(
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
      run(
        device,
        code(first),
        uniforms([geom.inner, len, geom.outer, len, 1, width]),
        bufs,
        groups
      );
      for (const b of temps) discard(b);
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
        run(
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
      run(
        device,
        local,
        u(nseq * nchunks),
        [out, buffer(args.src), totals],
        grid(nseq * nchunks).groups
      );
      if (nchunks > 1) {
        run(device, across, u(nseq), [totals], grid(nseq).groups);
        run(
          device,
          fix,
          u(nseq * args.len),
          [out, totals],
          grid(nseq * args.len).groups
        );
      }
      discard(totals);
    },

    /** Sort each sequence along the middle dimension of `src` viewed as
     *  [inner, len, outer]: the sequences are copied into work buffers of
     *  (value, position) pairs padded to a power of 2, sorted there by a
     *  bitonic network (a pass per stage), and copied out with `outIdx`
     *  the 1-based positions. Ties keep their order (positions compare),
     *  and NaN sorts last, or first with `descend`, as in MATLAB. */
    async sort(
      device: GPUDevice,
      args: {
        out: number;
        outIdx: number;
        src: number;
        inner: number;
        len: number;
        outer: number;
        descend: boolean;
      }
    ) {
      const { inner, len, outer } = args;
      const nseq = inner * outer;
      const out = create(device, args.out, nseq * len);
      const outIdx = args.outIdx
        ? create(device, args.outIdx, nseq * len)
        : null;
      if (nseq * len === 0) return;
      let pw = 1;
      while (pw < len) pw *= 2;
      const keys = device.createBuffer({ size: nseq * pw * 4, usage: usage() });
      const pos = device.createBuffer({ size: nseq * pw * 4, usage: usage() });
      const head = `
struct P { inner: u32, len: u32, pw: u32, nseq: u32, w: u32, k: u32, j: u32, desc: u32 }
@group(0) @binding(0) var<uniform> p: P;
fn at(s: u32, j: u32) -> u32 {
  return (s % p.inner) + j * p.inner + (s / p.inner) * p.inner * p.len;
}`;
      const u = (threads: number, k = 0, j = 0) =>
        uniforms([
          inner,
          len,
          pw,
          nseq,
          grid(threads).width,
          k,
          j,
          args.descend ? 1 : 0,
        ]);
      // Padding (position 0xffffffff) sorts after everything.
      const pack = `${head}
@group(0) @binding(1) var<storage, read_write> keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> pos: array<u32>;
@group(0) @binding(3) var<storage, read> src: array<f32>;
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let t = g.x + g.y * p.w;
  if (t >= p.nseq * p.pw) { return; }
  let s = t / p.pw;
  let j = t % p.pw;
  if (j < p.len) {
    keys[t] = src[at(s, j)];
    pos[t] = j;
  } else {
    keys[t] = 0.0;
    pos[t] = 0xffffffffu;
  }
}`;
      run(
        device,
        pack,
        u(nseq * pw),
        [keys, pos, buffer(args.src)],
        grid(nseq * pw).groups
      );
      // One bitonic stage: compare element i with i ^ j; blocks of size k
      // alternate direction.
      const stage = `${head}
@group(0) @binding(1) var<storage, read_write> keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> pos: array<u32>;
// Whether element x sorts before element y.
fn before(x: u32, y: u32) -> bool {
  let ix = pos[x];
  let iy = pos[y];
  if (ix == 0xffffffffu) { return false; }
  if (iy == 0xffffffffu) { return true; }
  let kx = keys[x];
  let ky = keys[y];
  let nx = isnan_(kx);
  let ny = isnan_(ky);
  if (nx != ny) { return select(ny, nx, p.desc != 0u); }
  if (!nx && kx != ky) { return select((kx < ky), (kx > ky), p.desc != 0u); }
  return ix < iy;
}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let t = g.x + g.y * p.w;
  if (t >= p.nseq * p.pw) { return; }
  let i = t % p.pw;
  let l = i ^ p.j;
  if (l <= i) { return; }
  let a = t;
  let b = t - i + l;
  let up = (i & p.k) == 0u;
  if (select(before(a, b), before(b, a), up)) {
    let ka = keys[a];
    keys[a] = keys[b];
    keys[b] = ka;
    let pa = pos[a];
    pos[a] = pos[b];
    pos[b] = pa;
  }
}`;
      // Stages comparing within blocks of `bs` elements (a block lies in
      // one sequence) run in workgroup memory, all of a block's stages for
      // k <= bs in one dispatch, and for larger k those with j < bs.
      // (Two elements per thread, within WebGPU's default 256 threads.)
      const bs = Math.min(pw, 512);
      const local = `${head}
@group(0) @binding(1) var<storage, read_write> keys: array<f32>;
@group(0) @binding(2) var<storage, read_write> pos: array<u32>;
var<workgroup> sk: array<f32, ${bs}>;
var<workgroup> sp: array<u32, ${bs}>;
fn before(x: u32, y: u32) -> bool {
  let ix = sp[x];
  let iy = sp[y];
  if (ix == 0xffffffffu) { return false; }
  if (iy == 0xffffffffu) { return true; }
  let kx = sk[x];
  let ky = sk[y];
  let nx = isnan_(kx);
  let ny = isnan_(ky);
  if (nx != ny) { return select(ny, nx, p.desc != 0u); }
  if (!nx && kx != ky) { return select((kx < ky), (kx > ky), p.desc != 0u); }
  return ix < iy;
}
@compute @workgroup_size(${bs / 2}) fn main(
  @builtin(local_invocation_index) t: u32,
  @builtin(workgroup_id) w: vec3u
) {
  let block = w.x + w.y * p.w;
  if (block * ${bs}u >= p.nseq * p.pw) { return; }
  let base = block * ${bs}u;
  sk[t] = keys[base + t];
  sp[t] = pos[base + t];
  sk[t + ${bs / 2}u] = keys[base + t + ${bs / 2}u];
  sp[t + ${bs / 2}u] = pos[base + t + ${bs / 2}u];
  workgroupBarrier();
  // k runs over [p.k, p.j] (powers of 2); j from min(k, bs) / 2 down to 1.
  for (var k = p.k; k <= p.j; k = k * 2u) {
    for (var j = min(k, ${bs}u) / 2u; j > 0u; j = j / 2u) {
      let i = 2u * j * (t / j) + (t % j);
      let up = (((base + i) % p.pw) & k) == 0u;
      if (select(before(i, i + j), before(i + j, i), up)) {
        let ka = sk[i];
        sk[i] = sk[i + j];
        sk[i + j] = ka;
        let pa = sp[i];
        sp[i] = sp[i + j];
        sp[i + j] = pa;
      }
      workgroupBarrier();
    }
  }
  keys[base + t] = sk[t];
  pos[base + t] = sp[t];
  keys[base + t + ${bs / 2}u] = sk[t + ${bs / 2}u];
  pos[base + t + ${bs / 2}u] = sp[t + ${bs / 2}u];
}`;
      const nblocks = (nseq * pw) / bs;
      const bx = Math.min(nblocks, 65535);
      const blocks: [number, number] = [bx, Math.ceil(nblocks / bx)];
      // For the local kernel, `k` and `j` hold the first and last k.
      const lu = (k0: number, k1: number) =>
        uniforms([inner, len, pw, nseq, bx, k0, k1, args.descend ? 1 : 0]);
      if (pw >= 2) run(device, local, lu(2, bs), [keys, pos], blocks);
      for (let k = 2 * bs; k <= pw; k *= 2) {
        for (let j = k >> 1; j >= bs; j >>= 1)
          run(
            device,
            stage,
            u(nseq * pw, k, j),
            [keys, pos],
            grid(nseq * pw).groups
          );
        run(device, local, lu(k, k), [keys, pos], blocks);
      }
      const unpack = `${head}
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> keys: array<f32>;
${outIdx ? "@group(0) @binding(3) var<storage, read> pos: array<u32>;" : ""}
${outIdx ? "@group(0) @binding(4) var<storage, read_write> dstIdx: array<f32>;" : ""}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let t = g.x + g.y * p.w;
  if (t >= p.nseq * p.len) { return; }
  let s = t / p.len;
  let j = t % p.len;
  dst[at(s, j)] = keys[s * p.pw + j];
  ${outIdx ? "dstIdx[at(s, j)] = f32(pos[s * p.pw + j] + 1u);" : ""}
}`;
      const bufs = [out, keys];
      if (outIdx) bufs.push(pos, outIdx);
      run(device, unpack, u(nseq * len), bufs, grid(nseq * len).groups);
      discard(keys);
      discard(pos);
    },

    /** Element `index` of buffer `id`, after the work that computes it. */
    async readAt(device: GPUDevice, args: { id: number; index: number }) {
      await settle(device);
      const staging = device.createBuffer({
        size: 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(buffer(args.id), args.index * 4, staging, 0, 4);
      device.queue.submit([enc.finish()]);
      await staging.mapAsync(GPUMapMode.READ);
      const v = new Float32Array(staging.getMappedRange())[0];
      staging.unmap();
      staging.destroy();
      return v;
    },

    /** Stream compaction (see `GpuKernels.compact`): each selected i
     *  (mask[i] != 0) with pos[i] <= count writes at pos[i] - 1 the value
     *  src[i], or (no `src`) i + 1, or (`m`) i % m + 1 and, into `out2`,
     *  floor(i / m) + 1. */
    async compact(
      device: GPUDevice,
      args: {
        out: number;
        out2: number;
        src: number;
        mask: number;
        pos: number;
        n: number;
        count: number;
        m: number;
      }
    ) {
      const out = create(device, args.out, args.count);
      const out2 = args.out2 ? create(device, args.out2, args.count) : null;
      if (args.count === 0) return;
      const value = args.src
        ? "src[i]"
        : args.m
          ? "f32(i % p.m + 1u)"
          : "f32(i + 1u)";
      const code = `
struct P { n: u32, w: u32, count: u32, m: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> mask: array<f32>;
@group(0) @binding(3) var<storage, read> pos: array<f32>;
${args.src ? "@group(0) @binding(4) var<storage, read> src: array<f32>;" : ""}
${out2 ? "@group(0) @binding(4) var<storage, read_write> dst2: array<f32>;" : ""}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + g.y * p.w;
  if (i >= p.n || mask[i] == 0.0) { return; }
  let q = u32(pos[i]);
  if (q > p.count) { return; }
  dst[q - 1u] = ${value};
  ${out2 ? "dst2[q - 1u] = f32(i / p.m + 1u);" : ""}
}`;
      const { groups, width } = grid(args.n);
      const bufs = [out, buffer(args.mask), buffer(args.pos)];
      if (args.src) bufs.push(buffer(args.src));
      if (out2) bufs.push(out2);
      run(
        device,
        code,
        uniforms([args.n, width, args.count, args.m]),
        bufs,
        groups
      );
    },

    /** out = src with each element selected by `mask` replaced by `rhs`: a
     *  scalar, a buffer of 1, or (with `pos`) rhs[pos[i] - 1]. */
    async expand(
      device: GPUDevice,
      args: {
        out: number;
        src: number;
        mask: number;
        pos: number;
        n: number;
        rhs: Operand;
      }
    ) {
      const out = create(device, args.out, args.n);
      if (args.n === 0) return;
      const rhs = args.rhs;
      const scalar = "scalar" in rhs;
      const usePos = !scalar && rhs.n !== 1 && args.pos !== 0;
      const value = scalar
        ? "p.s"
        : usePos
          ? "rhs[u32(pos[i]) - 1u]"
          : "rhs[0]";
      const code = `
struct P { n: u32, w: u32, p0: u32, p1: u32, s: f32, p2: f32, p3: f32, p4: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> dst: array<f32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read> mask: array<f32>;
${scalar ? "" : "@group(0) @binding(4) var<storage, read> rhs: array<f32>;"}
${usePos ? "@group(0) @binding(5) var<storage, read> pos: array<f32>;" : ""}
@compute @workgroup_size(256) fn main(@builtin(global_invocation_id) g: vec3u) {
  let i = g.x + g.y * p.w;
  if (i >= p.n) { return; }
  dst[i] = select(src[i], ${value}, mask[i] != 0.0);
}`;
      const { groups, width } = grid(args.n);
      const bufs = [out, buffer(args.src), buffer(args.mask)];
      if (!scalar) bufs.push(buffer(rhs.id));
      if (usePos) bufs.push(buffer(args.pos));
      run(
        device,
        code,
        uniforms([args.n, width, 0, 0], [scalar ? rhs.scalar : 0, 0, 0, 0]),
        bufs,
        groups
      );
    },

    /** out (m x n) = a (m x k) * b (k x n), column-major. A workgroup
     *  computes a tile of `bm` x `bn` outputs, each thread `tm` x `tn` of
     *  them (rows l.x + (bm/tm)*r, columns l.y + (bn/tn)*c), from slices of
     *  16 along k in workgroup memory. Large products use 64 x 64 tiles; a
     *  product with fewer than 48 of them uses 32 x 32, to fill the GPU. */
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
      if (args.k === 0) {
        // An empty inner dimension gives zeros.
        encoder(device).clearBuffer(out);
        return;
      }
      const large = Math.ceil(args.m / 64) * Math.ceil(args.n / 64) >= 48;
      const [bm, bn, tm, tn] = large ? [64, 64, 4, 4] : [32, 32, 2, 2];
      const bk = 16;
      const wx = bm / tm;
      const wy = bn / tn;
      const code = `
struct P { m: u32, k: u32, n: u32, pad: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> C: array<f32>;
@group(0) @binding(2) var<storage, read> A: array<f32>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
var<workgroup> As: array<f32, ${bk * bm}>;
var<workgroup> Bs: array<f32, ${bk * bn}>;
@compute @workgroup_size(${wx}, ${wy}) fn main(
  @builtin(local_invocation_id) l: vec3u,
  @builtin(local_invocation_index) t: u32,
  @builtin(workgroup_id) w: vec3u
) {
  let row0 = w.x * ${bm}u;
  let col0 = w.y * ${bn}u;
  var acc: array<array<f32, ${tn}>, ${tm}>;
  for (var k0 = 0u; k0 < p.k; k0 = k0 + ${bk}u) {
    // As[kk][ii] = A(row0 + ii, k0 + kk); Bs[kk][jj] = B(k0 + kk, col0 + jj),
    // 0 outside the matrices.
    for (var e = t; e < ${bk * bm}u; e = e + ${wx * wy}u) {
      let i = row0 + e % ${bm}u;
      let q = k0 + e / ${bm}u;
      var v: f32 = 0.0;
      if (i < p.m && q < p.k) { v = A[i + q * p.m]; }
      As[e] = v;
    }
    for (var e = t; e < ${bk * bn}u; e = e + ${wx * wy}u) {
      let kk = e % ${bk}u;
      let jj = e / ${bk}u;
      let q = k0 + kk;
      let j = col0 + jj;
      var v: f32 = 0.0;
      if (q < p.k && j < p.n) { v = B[q + j * p.k]; }
      Bs[kk * ${bn}u + jj] = v;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < ${bk}u; kk = kk + 1u) {
      var a: array<f32, ${tm}>;
      var b: array<f32, ${tn}>;
      for (var r = 0u; r < ${tm}u; r = r + 1u) { a[r] = As[kk * ${bm}u + l.x + ${wx}u * r]; }
      for (var c = 0u; c < ${tn}u; c = c + 1u) { b[c] = Bs[kk * ${bn}u + l.y + ${wy}u * c]; }
      for (var r = 0u; r < ${tm}u; r = r + 1u) {
        for (var c = 0u; c < ${tn}u; c = c + 1u) { acc[r][c] = acc[r][c] + a[r] * b[c]; }
      }
    }
    workgroupBarrier();
  }
  for (var r = 0u; r < ${tm}u; r = r + 1u) {
    let i = row0 + l.x + ${wx}u * r;
    if (i >= p.m) { continue; }
    for (var c = 0u; c < ${tn}u; c = c + 1u) {
      let j = col0 + l.y + ${wy}u * c;
      if (j < p.n) { C[i + j * p.m] = acc[r][c]; }
    }
  }
}`;
      run(
        device,
        code,
        uniforms([args.m, args.k, args.n, 0]),
        [out, buffer(args.a), buffer(args.b)],
        [Math.ceil(args.m / bm), Math.ceil(args.n / bn)]
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
      run(
        device,
        code,
        uniforms([args.m, args.n, width, 0]),
        [out, buffer(args.src)],
        groups
      );
    },

    async synchronize(device: GPUDevice) {
      await settle(device);
      await device.queue.onSubmittedWorkDone();
    },

    dispose() {
      enc = null;
      pass = null;
      dispatches = 0;
      doomed.length = 0;
      scopes = [];
      uniBuf?.destroy();
      uniBuf = null;
      for (const b of buffers.values()) b.destroy();
      buffers.clear();
      for (const list of pool.values()) for (const b of list) b.destroy();
      pool.clear();
      for (const b of tables.values()) b.destroy();
      tables.clear();
      pooled = 0;
    },
  };
}
