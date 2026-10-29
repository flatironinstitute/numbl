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
      const b = device.createBuffer({
        size: Math.max(4, args.data.byteLength),
        usage: usage(),
      });
      if (args.data.byteLength > 0) device.queue.writeBuffer(b, 0, args.data);
      buffers.set(args.id, b);
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
        buffers.get(id)?.destroy();
        buffers.delete(id);
      }
    },

    async synchronize(device: GPUDevice) {
      await device.queue.onSubmittedWorkDone();
    },

    dispose() {
      for (const b of buffers.values()) b.destroy();
      buffers.clear();
    },
  };
}
