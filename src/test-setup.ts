/**
 * Vitest setup: install the qhull Delaunay backend before any test runs.
 *
 * Unit tests exercise delaunay/delaunayn through executeCode without going via
 * the CLI / worker / test-runner startup that normally installs the backend,
 * so we install it here (the Node loader reads the WASM from disk).
 */
import { beforeAll } from "vitest";
import { loadQhullNodeBackend } from "./numbl-core/native/qhull-node.js";

beforeAll(async () => {
  await loadQhullNodeBackend();
});
