import Dexie, { type EntityTable } from "dexie";

export interface Project {
  name: string; // Primary key - unique project name (no spaces)
  displayName?: string; // Optional display name (can have spaces)
  createdAt: number; // Timestamp
  updatedAt: number; // Timestamp
  lastOpenedAt: number; // For sorting by recent use
}

export interface ProjectFile {
  id: string; // Primary key - UUID
  projectName: string; // Foreign key to Project.name (indexed)
  path: string; // File path like "src/utils/helper.m"
  content: string; // File content
  createdAt: number; // Timestamp
  updatedAt: number; // Timestamp
}

export interface MipPackageCache {
  name: string; // Primary key - package name
  version: string; // For cache invalidation against index
  files: { path: string; source: string; data?: Uint8Array }[]; // Extracted files (.m, .js, .wasm)
  loadPaths: string[]; // Resolved addpath paths from load_package.m
  fetchedAt: number; // Timestamp
}

export class NumblDatabase extends Dexie {
  projects!: EntityTable<Project, "name">;
  files!: EntityTable<ProjectFile, "id">;
  mipPackages!: EntityTable<MipPackageCache, "name">;

  constructor() {
    super("numbl-db");

    this.version(1).stores({
      projects: "name, lastOpenedAt",
      files: "id, projectName, [projectName+path]",
    });

    this.version(2).stores({
      projects: "name, lastOpenedAt",
      files: "id, projectName, [projectName+path]",
      mipPackages: "name",
    });
  }
}

export const db = new NumblDatabase();
