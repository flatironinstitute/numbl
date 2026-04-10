import { useRef, useEffect, type CSSProperties } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { Line2 } from "three/examples/jsm/lines/Line2.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { LineGeometry } from "three/examples/jsm/lines/LineGeometry.js";
import type { SurfTrace, Plot3Trace, Bar3Trace } from "./types.js";
import { colormapLookup } from "./surfColormap.js";

// Color order for plot3 traces
const TRACE_COLORS = [
  [0, 0.447, 0.741], // #0072BD blue
  [0.85, 0.325, 0.098], // #D95319 red-orange
  [0.929, 0.694, 0.125], // #EDB120 yellow
  [0.494, 0.184, 0.556], // #7E2F8E purple
  [0.466, 0.674, 0.188], // #77AC30 green
  [0.301, 0.745, 0.933], // #4DBEEE cyan
  [0.635, 0.078, 0.184], // #A2142F dark red
];

interface SurfViewProps {
  surfTraces: SurfTrace[];
  plot3Traces?: Plot3Trace[];
  bar3Traces?: Bar3Trace[];
  bar3hTraces?: Bar3Trace[];
  shading?: "faceted" | "flat" | "interp";
  colorbar?: boolean;
  colorbarLocation?: string;
  colormap?: string;
}

export function SurfView({
  surfTraces,
  plot3Traces = [],
  bar3Traces = [],
  bar3hTraces = [],
  shading,
  colorbar,
  colorbarLocation,
  colormap,
}: SurfViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef<{
    renderer: THREE.WebGLRenderer;
    scene: THREE.Scene;
    camera: THREE.OrthographicCamera;
    controls: OrbitControls;
    animId: number;
  } | null>(null);

  // Set up the three.js scene once
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setClearColor(0xffffff);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();

    // Orthographic camera — frustum will be sized on resize
    const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 100);
    camera.position.set(1.2, 0.8, 1.2);
    camera.lookAt(0, 0, 0);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enablePan = false;
    controls.enableZoom = true;

    // Ambient + directional light
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
    dirLight.position.set(2, 3, 2);
    scene.add(dirLight);

    const animId = requestAnimationFrame(function loop() {
      controls.update();
      renderer.render(scene, camera);
      stateRef.current!.animId = requestAnimationFrame(loop);
    });

    stateRef.current = { renderer, scene, camera, controls, animId };

    // Handle resize
    const observer = new ResizeObserver(() => {
      const rect = container.getBoundingClientRect();
      if (rect.width === 0 || rect.height === 0) return;
      renderer.setSize(rect.width, rect.height);
      const aspect = rect.width / rect.height;
      const frustumSize = 1.2;
      camera.left = -frustumSize * aspect;
      camera.right = frustumSize * aspect;
      camera.top = frustumSize;
      camera.bottom = -frustumSize;
      camera.updateProjectionMatrix();
    });
    observer.observe(container);

    return () => {
      observer.disconnect();
      cancelAnimationFrame(stateRef.current?.animId ?? animId);
      controls.dispose();
      renderer.dispose();
      container.removeChild(renderer.domElement);
      stateRef.current = null;
    };
  }, []);

  // Rebuild scene when data changes
  useEffect(() => {
    const st = stateRef.current;
    if (!st) return;
    const { scene } = st;

    // Remove old meshes/lines (keep lights)
    const toRemove: THREE.Object3D[] = [];
    scene.traverse(obj => {
      if (
        obj instanceof THREE.Mesh ||
        obj instanceof THREE.LineSegments ||
        obj instanceof THREE.Line
      ) {
        toRemove.push(obj);
      }
    });
    for (const obj of toRemove) {
      scene.remove(obj);
      if ((obj as THREE.Mesh).geometry) (obj as THREE.Mesh).geometry.dispose();
    }

    if (
      surfTraces.length === 0 &&
      plot3Traces.length === 0 &&
      bar3Traces.length === 0 &&
      bar3hTraces.length === 0
    )
      return;

    // Compute global data ranges across both surf and plot3 traces
    let xMin = Infinity,
      xMax = -Infinity;
    let yMin = Infinity,
      yMax = -Infinity;
    let zMin = Infinity,
      zMax = -Infinity;

    const updateRange = (
      arr: number[],
      updateMin: { v: number },
      updateMax: { v: number }
    ) => {
      for (const v of arr) {
        if (isFinite(v)) {
          if (v < updateMin.v) updateMin.v = v;
          if (v > updateMax.v) updateMax.v = v;
        }
      }
    };

    const xMinRef = { v: xMin },
      xMaxRef = { v: xMax };
    const yMinRef = { v: yMin },
      yMaxRef = { v: yMax };
    const zMinRef = { v: zMin },
      zMaxRef = { v: zMax };

    for (const trace of surfTraces) {
      updateRange(trace.x, xMinRef, xMaxRef);
      updateRange(trace.y, yMinRef, yMaxRef);
      updateRange(trace.z, zMinRef, zMaxRef);
    }
    for (const trace of plot3Traces) {
      updateRange(trace.x, xMinRef, xMaxRef);
      updateRange(trace.y, yMinRef, yMaxRef);
      updateRange(trace.z, zMinRef, zMaxRef);
    }
    for (const trace of bar3Traces) {
      updateRange(trace.x, xMinRef, xMaxRef);
      updateRange(trace.y, yMinRef, yMaxRef);
      updateRange(trace.z, zMinRef, zMaxRef);
      // Bars extend to zero on z-axis
      if (0 < zMinRef.v) zMinRef.v = 0;
    }
    for (const trace of bar3hTraces) {
      // bar3h: bars extend along x-axis, positions on y and z axes
      updateRange(trace.y, yMinRef, yMaxRef);
      updateRange(trace.z, zMinRef, zMaxRef);
      updateRange(trace.x, xMinRef, xMaxRef);
      // Bars extend to zero on x-axis
      if (0 < xMinRef.v) xMinRef.v = 0;
    }

    xMin = xMinRef.v;
    xMax = xMaxRef.v;
    yMin = yMinRef.v;
    yMax = yMaxRef.v;
    zMin = zMinRef.v;
    zMax = zMaxRef.v;

    if (!isFinite(xMin)) return;
    if (xMax === xMin) {
      xMin -= 1;
      xMax += 1;
    }
    if (yMax === yMin) {
      yMin -= 1;
      yMax += 1;
    }
    if (zMax === zMin) {
      zMin -= 1;
      zMax += 1;
    }

    const xRange = xMax - xMin || 1;
    const yRange = yMax - yMin || 1;
    const zRange2 = zMax - zMin || 1;
    const rangeMax = Math.max(xRange, yRange, zRange2);
    const cxData = (xMin + xMax) / 2;
    const cyData = (yMin + yMax) / 2;
    const czData = (zMin + zMax) / 2;

    // For bar3/bar3h: use per-axis scaling when z range dominates x/y range.
    // This prevents bars from appearing as thin sticks in histogram2-style data.
    const hasOnlyBars =
      surfTraces.length === 0 &&
      plot3Traces.length === 0 &&
      (bar3Traces.length > 0 || bar3hTraces.length > 0);
    const barRangeMax = hasOnlyBars ? Math.max(xRange, yRange) : rangeMax;
    // normBar scales x/y to fill the view; normZ still uses rangeMax for z
    const normBar = (v: number, center: number) => (v - center) / barRangeMax;
    const normBarZ = (v: number, center: number) =>
      (v - center) / (hasOnlyBars ? Math.max(barRangeMax, zRange2) : rangeMax);

    // Normalize a data point to [-0.5, 0.5] range
    const norm = (v: number, center: number) => (v - center) / rangeMax;

    // ── Render surf traces ──────────────────────────────────────────────
    for (const trace of surfTraces) {
      const { rows, cols, x, y, z } = trace;
      const zRange = zMax - zMin || 1;
      const alpha = trace.faceAlpha ?? 1;

      // Build indexed geometry
      const positions = new Float32Array(rows * cols * 3);
      const colors = new Float32Array(rows * cols * 3);

      for (let j = 0; j < cols; j++) {
        for (let i = 0; i < rows; i++) {
          const idx = j * rows + i; // column-major
          const vi = i * cols + j; // vertex index for buffer (row-major)

          const nx = norm(x[idx], cxData);
          const ny = norm(y[idx], cyData);
          const nz = norm(z[idx], czData);

          // three.js: X=right, Y=up, Z=towards camera
          // Map data X→three X, data Y→three Z, data Z→three Y
          positions[vi * 3] = nx;
          positions[vi * 3 + 1] = nz;
          positions[vi * 3 + 2] = ny;

          const t = trace.c
            ? (trace.c[idx] - zMin) / zRange
            : (z[idx] - zMin) / zRange;
          const [r, g, b] = colormapLookup(t);
          colors[vi * 3] = r;
          colors[vi * 3 + 1] = g;
          colors[vi * 3 + 2] = b;
        }
      }

      // Triangle indices
      const indices: number[] = [];
      for (let i = 0; i < rows - 1; i++) {
        for (let j = 0; j < cols - 1; j++) {
          const a = i * cols + j;
          const b = i * cols + (j + 1);
          const c = (i + 1) * cols + j;
          const d = (i + 1) * cols + (j + 1);
          indices.push(a, c, b);
          indices.push(b, c, d);
        }
      }

      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute(
        "position",
        new THREE.BufferAttribute(positions, 3)
      );
      geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
      geometry.setIndex(indices);
      geometry.computeVertexNormals();

      // Determine effective shading mode
      const shadingMode = shading ?? "faceted";
      const useFlat = shadingMode === "faceted" || shadingMode === "flat";

      // Face material
      const showFaces = trace.faceColor !== "none";
      if (showFaces) {
        let faceMaterial: THREE.Material;
        if (Array.isArray(trace.faceColor)) {
          const [r, g, b] = trace.faceColor;
          faceMaterial = new THREE.MeshPhongMaterial({
            color: new THREE.Color(r, g, b),
            flatShading: useFlat,
            opacity: alpha,
            transparent: alpha < 1,
            side: THREE.DoubleSide,
          });
        } else {
          faceMaterial = new THREE.MeshPhongMaterial({
            vertexColors: true,
            flatShading: useFlat,
            opacity: alpha,
            transparent: alpha < 1,
            side: THREE.DoubleSide,
          });
        }
        scene.add(new THREE.Mesh(geometry, faceMaterial));
      }

      // Edge wireframe — hidden for "flat" and "interp" shading modes
      const showEdges = trace.edgeColor !== "none" && shadingMode === "faceted";
      if (showEdges) {
        const edgePositions: number[] = [];
        const edgeColors: number[] = [];
        for (let i = 0; i < rows; i++) {
          for (let j = 0; j < cols; j++) {
            const vi = i * cols + j;
            // Horizontal edge (to the right)
            if (j < cols - 1) {
              const vi2 = i * cols + (j + 1);
              edgePositions.push(
                positions[vi * 3],
                positions[vi * 3 + 1],
                positions[vi * 3 + 2],
                positions[vi2 * 3],
                positions[vi2 * 3 + 1],
                positions[vi2 * 3 + 2]
              );
              edgeColors.push(
                colors[vi * 3],
                colors[vi * 3 + 1],
                colors[vi * 3 + 2],
                colors[vi2 * 3],
                colors[vi2 * 3 + 1],
                colors[vi2 * 3 + 2]
              );
            }
            // Vertical edge (downward)
            if (i < rows - 1) {
              const vi2 = (i + 1) * cols + j;
              edgePositions.push(
                positions[vi * 3],
                positions[vi * 3 + 1],
                positions[vi * 3 + 2],
                positions[vi2 * 3],
                positions[vi2 * 3 + 1],
                positions[vi2 * 3 + 2]
              );
              edgeColors.push(
                colors[vi * 3],
                colors[vi * 3 + 1],
                colors[vi * 3 + 2],
                colors[vi2 * 3],
                colors[vi2 * 3 + 1],
                colors[vi2 * 3 + 2]
              );
            }
          }
        }

        const edgeGeometry = new THREE.BufferGeometry();
        edgeGeometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(edgePositions, 3)
        );

        let edgeMat: THREE.LineBasicMaterial;
        if (Array.isArray(trace.edgeColor)) {
          const [r, g, b] = trace.edgeColor;
          edgeMat = new THREE.LineBasicMaterial({
            color: new THREE.Color(r, g, b),
          });
        } else {
          edgeMat = new THREE.LineBasicMaterial({
            color: 0x000000,
            opacity: 0.3,
            transparent: true,
          });
        }
        scene.add(new THREE.LineSegments(edgeGeometry, edgeMat));
      }
    }

    // ── Render plot3 traces ─────────────────────────────────────────────
    for (let ti = 0; ti < plot3Traces.length; ti++) {
      const trace = plot3Traces[ti];
      const { x, y, z } = trace;

      // Determine color
      const defaultColor = TRACE_COLORS[ti % TRACE_COLORS.length];
      const color = trace.color ?? defaultColor;
      const threeColor = new THREE.Color(color[0], color[1], color[2]);

      // Build line points (skip NaN/Inf to create line breaks)
      const showLine = trace.lineStyle !== "none";
      if (showLine) {
        // Build segments of consecutive finite points
        const segments: THREE.Vector3[][] = [];
        let currentSegment: THREE.Vector3[] = [];

        for (let i = 0; i < x.length; i++) {
          if (isFinite(x[i]) && isFinite(y[i]) && isFinite(z[i])) {
            const nx = norm(x[i], cxData);
            const ny = norm(y[i], cyData);
            const nz = norm(z[i], czData);
            // Map: data X→three X, data Z→three Y, data Y→three Z
            currentSegment.push(new THREE.Vector3(nx, nz, ny));
          } else {
            if (currentSegment.length > 0) {
              segments.push(currentSegment);
              currentSegment = [];
            }
          }
        }
        if (currentSegment.length > 0) {
          segments.push(currentSegment);
        }

        // Use Line2 + LineMaterial for proper line width support
        // (THREE.LineBasicMaterial.linewidth is ignored on most platforms)
        const lw = trace.lineWidth ?? 2;
        const isDashed =
          trace.lineStyle === "--" ||
          trace.lineStyle === ":" ||
          trace.lineStyle === "-.";

        for (const seg of segments) {
          if (seg.length < 2) continue;
          const positions: number[] = [];
          for (const pt of seg) {
            positions.push(pt.x, pt.y, pt.z);
          }

          if (isDashed) {
            // Fall back to LineDashedMaterial for dash patterns
            // (Line2/LineMaterial doesn't support dashes)
            const dashedMat = new THREE.LineDashedMaterial({
              color: threeColor,
              linewidth: lw,
              dashSize: trace.lineStyle === ":" ? 0.01 : 0.03,
              gapSize: trace.lineStyle === ":" ? 0.02 : 0.015,
            });
            const geo = new THREE.BufferGeometry().setFromPoints(seg);
            const line = new THREE.Line(geo, dashedMat);
            line.computeLineDistances();
            scene.add(line);
          } else {
            const geo = new LineGeometry();
            geo.setPositions(positions);
            const mat = new LineMaterial({
              color: threeColor.getHex(),
              linewidth: lw,
              worldUnits: false,
              resolution: new THREE.Vector2(
                st.renderer.domElement.width || 800,
                st.renderer.domElement.height || 600
              ),
            });
            scene.add(new Line2(geo, mat));
          }
        }
      }

      // Draw markers as small spheres/points
      if (trace.marker && trace.marker !== "none") {
        const markerSize = (trace.markerSize ?? 6) / 600; // scale to normalized space
        const markerColor = trace.markerEdgeColor
          ? new THREE.Color(
              trace.markerEdgeColor[0],
              trace.markerEdgeColor[1],
              trace.markerEdgeColor[2]
            )
          : threeColor;

        const indices = trace.markerIndices
          ? trace.markerIndices.map(i => i - 1) // 1-based
          : Array.from({ length: x.length }, (_, i) => i);

        const markerGeo = new THREE.SphereGeometry(markerSize, 8, 8);
        const markerMat = new THREE.MeshBasicMaterial({ color: markerColor });

        for (const i of indices) {
          if (i < 0 || i >= x.length) continue;
          if (!isFinite(x[i]) || !isFinite(y[i]) || !isFinite(z[i])) continue;
          const nx = norm(x[i], cxData);
          const ny = norm(y[i], cyData);
          const nz = norm(z[i], czData);
          const mesh = new THREE.Mesh(markerGeo, markerMat);
          mesh.position.set(nx, nz, ny);
          scene.add(mesh);
        }
      }
    }

    // ── Render bar3 traces (vertical 3D bars) ────────────────────────────
    for (const trace of bar3Traces) {
      const halfW = (trace.width / 2) * 0.9; // slight shrink to show gaps
      const zRangeT = zMax - zMin || 1;
      for (let i = 0; i < trace.x.length; i++) {
        const bx = trace.x[i];
        const by = trace.y[i];
        const bz = trace.z[i];
        if (!isFinite(bz)) continue;

        const barHeight = Math.abs(normBarZ(bz, czData) - normBarZ(0, czData));
        const barCenter = (normBarZ(bz, czData) + normBarZ(0, czData)) / 2;

        const geo = new THREE.BoxGeometry(
          (halfW * 2) / barRangeMax,
          barHeight,
          (halfW * 2) / barRangeMax
        );

        const t = (bz - zMin) / zRangeT;
        const [cr, cg, cb] = trace.color ?? colormapLookup(t);
        const mat = new THREE.MeshPhongMaterial({
          color: new THREE.Color(cr, cg, cb),
        });
        const mesh = new THREE.Mesh(geo, mat);
        // data X→three X, data Z→three Y, data Y→three Z
        mesh.position.set(normBar(bx, cxData), barCenter, normBar(by, cyData));
        scene.add(mesh);

        // Edge wireframe
        const edges = new THREE.EdgesGeometry(geo);
        const lineMat = new THREE.LineBasicMaterial({
          color: 0x000000,
          opacity: 0.3,
          transparent: true,
        });
        const wireframe = new THREE.LineSegments(edges, lineMat);
        wireframe.position.copy(mesh.position);
        scene.add(wireframe);
      }
    }

    // ── Render bar3h traces (horizontal 3D bars) ───────────────────────
    for (const trace of bar3hTraces) {
      const halfW = (trace.width / 2) * 0.9;
      const xRangeH = xMax - xMin || 1;
      // bar3h: x=positions (category axis, mapped to z-axis in MATLAB),
      //        y=bar lengths (value axis, mapped to y/horizontal),
      //        z values are the bar lengths, x values are positions
      // Reinterpret: y-positions on z-axis, x-values are bar lengths on x-axis
      for (let i = 0; i < trace.x.length; i++) {
        const pos = trace.y[i]; // position on y-axis
        const colIdx = trace.x[i]; // position on x-axis (column)
        const len = trace.z[i]; // bar length along x-axis
        if (!isFinite(len)) continue;

        const barLength = Math.abs(normBar(len, cxData) - normBar(0, cxData));
        const barCenter = (normBar(len, cxData) + normBar(0, cxData)) / 2;

        const geo = new THREE.BoxGeometry(
          barLength,
          (halfW * 2) / barRangeMax,
          (halfW * 2) / barRangeMax
        );

        const t = (len - xMin) / xRangeH;
        const [cr, cg, cb] = trace.color ?? colormapLookup(t);
        const mat = new THREE.MeshPhongMaterial({
          color: new THREE.Color(cr, cg, cb),
        });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.position.set(
          barCenter,
          normBar(colIdx, czData),
          normBar(pos, cyData)
        );
        scene.add(mesh);

        const edges = new THREE.EdgesGeometry(geo);
        const lineMat = new THREE.LineBasicMaterial({
          color: 0x000000,
          opacity: 0.3,
          transparent: true,
        });
        const wireframe = new THREE.LineSegments(edges, lineMat);
        wireframe.position.copy(mesh.position);
        scene.add(wireframe);
      }
    }

    // Axis lines
    addAxisLines(
      scene,
      xMin,
      xMax,
      yMin,
      yMax,
      zMin,
      zMax,
      rangeMax,
      cxData,
      cyData,
      czData
    );
  }, [surfTraces, plot3Traces, bar3Traces, bar3hTraces, shading]);

  // Compute color range for the colorbar from surf traces (uses C if present,
  // otherwise Z). Falls back to bar3 z values when no surf traces are present.
  let cbMin = Infinity;
  let cbMax = -Infinity;
  for (const t of surfTraces) {
    const arr = t.c ?? t.z;
    for (const v of arr) {
      if (isFinite(v)) {
        if (v < cbMin) cbMin = v;
        if (v > cbMax) cbMax = v;
      }
    }
  }
  if (!isFinite(cbMin)) {
    for (const t of bar3Traces) {
      for (const v of t.z) {
        if (isFinite(v)) {
          if (v < cbMin) cbMin = v;
          if (v > cbMax) cbMax = v;
        }
      }
    }
  }
  const haveColorRange = isFinite(cbMin) && isFinite(cbMax);
  if (cbMin === cbMax) {
    cbMin -= 0.5;
    cbMax += 0.5;
  }

  return (
    <div style={{ position: "relative", width: "100%", height: "100%" }}>
      <div ref={containerRef} style={{ position: "absolute", inset: 0 }} />
      {colorbar && haveColorRange && (
        <ColorbarOverlay
          location={(colorbarLocation ?? "eastoutside").toLowerCase()}
          dMin={cbMin}
          dMax={cbMax}
          colormap={colormap}
        />
      )}
    </div>
  );
}

// ── Colorbar overlay (HTML, drawn on top of the Three.js canvas) ────────

function ColorbarOverlay({
  location,
  dMin,
  dMax,
  colormap,
}: {
  location: string;
  dMin: number;
  dMax: number;
  colormap?: string;
}) {
  // Build a CSS gradient from N samples of the colormap.
  // (colormap name is currently unused — surfColormap.colormapLookup uses parula.)
  void colormap;
  const N = 32;
  const stops: string[] = [];
  for (let i = 0; i < N; i++) {
    const t = i / (N - 1);
    const [r, g, b] = colormapLookup(t);
    const rgb = `rgb(${Math.round(r * 255)},${Math.round(g * 255)},${Math.round(b * 255)})`;
    stops.push(`${rgb} ${(t * 100).toFixed(2)}%`);
  }
  const horizontal =
    location === "northoutside" ||
    location === "southoutside" ||
    location === "north" ||
    location === "south";
  // Vertical gradients go bottom→top so the max sits at the top.
  const gradient = horizontal
    ? `linear-gradient(to right, ${stops.join(",")})`
    : `linear-gradient(to top, ${stops.join(",")})`;

  const fmt = (v: number) =>
    Number.isInteger(v) ? String(v) : v.toPrecision(3);

  // Position styles per location
  const barThickness = 16;
  const containerStyle: CSSProperties = {
    position: "absolute",
    pointerEvents: "none",
    fontFamily: "sans-serif",
    fontSize: 10,
    color: "#333",
  };

  const barStyle: CSSProperties = {
    background: gradient,
    border: "1px solid #999",
    boxSizing: "border-box",
  };

  switch (location) {
    case "eastoutside":
      return (
        <div
          style={{
            ...containerStyle,
            top: 12,
            bottom: 12,
            right: 8,
            width: 50,
            display: "flex",
            alignItems: "stretch",
          }}
        >
          <div style={{ ...barStyle, width: barThickness, height: "100%" }} />
          <div
            style={{
              marginLeft: 4,
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
            }}
          >
            <span>{fmt(dMax)}</span>
            <span>{fmt(dMin)}</span>
          </div>
        </div>
      );
    case "westoutside":
      return (
        <div
          style={{
            ...containerStyle,
            top: 12,
            bottom: 12,
            left: 8,
            width: 50,
            display: "flex",
            alignItems: "stretch",
            flexDirection: "row-reverse",
          }}
        >
          <div style={{ ...barStyle, width: barThickness, height: "100%" }} />
          <div
            style={{
              marginRight: 4,
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
              textAlign: "right",
            }}
          >
            <span>{fmt(dMax)}</span>
            <span>{fmt(dMin)}</span>
          </div>
        </div>
      );
    case "northoutside":
      return (
        <div
          style={{
            ...containerStyle,
            left: 12,
            right: 12,
            top: 8,
            height: 32,
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              marginBottom: 2,
            }}
          >
            <span>{fmt(dMin)}</span>
            <span>{fmt(dMax)}</span>
          </div>
          <div style={{ ...barStyle, height: barThickness, width: "100%" }} />
        </div>
      );
    case "southoutside":
      return (
        <div
          style={{
            ...containerStyle,
            left: 12,
            right: 12,
            bottom: 8,
            height: 32,
            display: "flex",
            flexDirection: "column-reverse",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              marginTop: 2,
            }}
          >
            <span>{fmt(dMin)}</span>
            <span>{fmt(dMax)}</span>
          </div>
          <div style={{ ...barStyle, height: barThickness, width: "100%" }} />
        </div>
      );
    case "east":
      return (
        <div
          style={{
            ...containerStyle,
            top: 24,
            bottom: 24,
            right: 24,
            width: 50,
            display: "flex",
            flexDirection: "row-reverse",
            alignItems: "stretch",
          }}
        >
          <div style={{ ...barStyle, width: barThickness, height: "100%" }} />
          <div
            style={{
              marginRight: 4,
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
              textAlign: "right",
            }}
          >
            <span>{fmt(dMax)}</span>
            <span>{fmt(dMin)}</span>
          </div>
        </div>
      );
    case "west":
      return (
        <div
          style={{
            ...containerStyle,
            top: 24,
            bottom: 24,
            left: 24,
            width: 50,
            display: "flex",
            alignItems: "stretch",
          }}
        >
          <div style={{ ...barStyle, width: barThickness, height: "100%" }} />
          <div
            style={{
              marginLeft: 4,
              display: "flex",
              flexDirection: "column",
              justifyContent: "space-between",
            }}
          >
            <span>{fmt(dMax)}</span>
            <span>{fmt(dMin)}</span>
          </div>
        </div>
      );
    case "north":
      return (
        <div
          style={{
            ...containerStyle,
            left: 24,
            right: 24,
            top: 24,
            height: 32,
            display: "flex",
            flexDirection: "column-reverse",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              marginTop: 2,
            }}
          >
            <span>{fmt(dMin)}</span>
            <span>{fmt(dMax)}</span>
          </div>
          <div style={{ ...barStyle, height: barThickness, width: "100%" }} />
        </div>
      );
    case "south":
      return (
        <div
          style={{
            ...containerStyle,
            left: 24,
            right: 24,
            bottom: 24,
            height: 32,
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              marginBottom: 2,
            }}
          >
            <span>{fmt(dMin)}</span>
            <span>{fmt(dMax)}</span>
          </div>
          <div style={{ ...barStyle, height: barThickness, width: "100%" }} />
        </div>
      );
    default:
      return null;
  }
}

function addAxisLines(
  scene: THREE.Scene,
  xMin: number,
  xMax: number,
  yMin: number,
  yMax: number,
  zMin: number,
  zMax: number,
  rangeMax: number,
  cxData: number,
  cyData: number,
  czData: number
) {
  const norm = (v: number, center: number) => (v - center) / rangeMax;

  const axes: {
    from: [number, number, number];
    to: [number, number, number];
  }[] = [
    { from: [xMin, yMin, zMin], to: [xMax, yMin, zMin] },
    { from: [xMin, yMin, zMin], to: [xMin, yMax, zMin] },
    { from: [xMin, yMin, zMin], to: [xMin, yMin, zMax] },
  ];

  const mat = new THREE.LineBasicMaterial({ color: 0x333333 });

  for (const axis of axes) {
    const pts = [axis.from, axis.to].map(([ax, ay, az]) => {
      const nx = norm(ax, cxData);
      const ny = norm(ay, cyData);
      const nz = norm(az, czData);
      return new THREE.Vector3(nx, nz, ny); // data X→X, data Z→Y, data Y→Z
    });
    const geo = new THREE.BufferGeometry().setFromPoints(pts);
    scene.add(new THREE.Line(geo, mat));
  }
}
