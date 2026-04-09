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
  createdAt: number; // Timestamp
  updatedAt: number; // Timestamp
}

export interface FileContent {
  id: string; // Same as ProjectFile.id
  data: Uint8Array; // File content (binary)
}

export class NumblDatabase extends Dexie {
  projects!: EntityTable<Project, "name">;
  files!: EntityTable<ProjectFile, "id">;
  fileContents!: EntityTable<FileContent, "id">;

  constructor() {
    super("numbl-db");

    this.version(1).stores({
      projects: "name, lastOpenedAt",
      files: "id, projectName, [projectName+path]",
    });

    this.version(2).stores({
      projects: "name, lastOpenedAt",
      files: "id, projectName, [projectName+path]",
    });

    this.version(3)
      .stores({
        projects: "name, lastOpenedAt",
        files: "id, projectName, [projectName+path]",
        fileContents: "id",
      })
      .upgrade(async tx => {
        // Migrate data from files table to fileContents table
        const filesTable = tx.table("files");
        const contentsTable = tx.table("fileContents");
        const allFiles = await filesTable.toArray();
        const contentRecords: FileContent[] = [];
        for (const f of allFiles) {
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          const data = (f as any).data;
          if (data) {
            contentRecords.push({ id: f.id, data });
          }
        }
        if (contentRecords.length > 0) {
          await contentsTable.bulkAdd(contentRecords);
        }
        // Remove data field from files records
        await filesTable.toCollection().modify((f: Record<string, unknown>) => {
          delete f.data;
        });
      });

    // v4: rename the shared __home__ project to __system__
    this.version(4)
      .stores({
        projects: "name, lastOpenedAt",
        files: "id, projectName, [projectName+path]",
        fileContents: "id",
      })
      .upgrade(async tx => {
        const projectsTable = tx.table("projects");
        const filesTable = tx.table("files");
        const oldProject = await projectsTable.get("__home__");
        if (oldProject) {
          await projectsTable.add({ ...oldProject, name: "__system__" });
          await projectsTable.delete("__home__");
        }
        await filesTable
          .where("projectName")
          .equals("__home__")
          .modify({ projectName: "__system__" });
      });
  }
}

export const db = new NumblDatabase();
