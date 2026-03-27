import { db, type Project, type ProjectFile } from "./schema";

const textEncoder = new TextEncoder();

const DEFAULT_CODE = `% Write your script here
tic;
a = 0;

for i = 1:1e7
    a = a + i;
end

fprintf('Sum: %d\\n', a);
fprintf('Elapsed: %f sec\\n', toc);
`;

// Project Operations

export async function createProject(name: string): Promise<void> {
  const now = Date.now();

  await db.transaction("rw", db.projects, db.files, async () => {
    // Create project
    await db.projects.add({
      name,
      createdAt: now,
      updatedAt: now,
      lastOpenedAt: now,
    });

    // Create default file
    await db.files.add({
      id: crypto.randomUUID(),
      projectName: name,
      path: "script.m",
      data: textEncoder.encode(DEFAULT_CODE),
      createdAt: now,
      updatedAt: now,
    });
  });
}

export async function getProject(name: string): Promise<Project | undefined> {
  return await db.projects.get(name);
}

export async function listProjects(): Promise<Project[]> {
  return await db.projects.orderBy("lastOpenedAt").reverse().toArray();
}

export async function deleteProject(name: string): Promise<void> {
  await db.transaction("rw", db.projects, db.files, async () => {
    // Delete all files in project
    await db.files.where("projectName").equals(name).delete();
    // Delete project
    await db.projects.delete(name);
  });
}

export async function renameProject(
  oldName: string,
  newName: string
): Promise<void> {
  await db.transaction("rw", db.projects, db.files, async () => {
    const project = await db.projects.get(oldName);
    if (!project) {
      throw new Error("Project not found");
    }

    // Update all files to reference new project name
    await db.files
      .where("projectName")
      .equals(oldName)
      .modify({ projectName: newName });

    // Delete old project and create new one with same data
    await db.projects.delete(oldName);
    await db.projects.add({
      ...project,
      name: newName,
      updatedAt: Date.now(),
    });
  });
}

export async function updateLastOpened(projectName: string): Promise<void> {
  await db.projects.update(projectName, { lastOpenedAt: Date.now() });
}

// File Operations

export async function getProjectFiles(
  projectName: string
): Promise<ProjectFile[]> {
  return await db.files.where("projectName").equals(projectName).toArray();
}

export async function getFile(
  fileId: string
): Promise<ProjectFile | undefined> {
  return await db.files.get(fileId);
}

export async function saveFileData(
  fileId: string,
  data: Uint8Array
): Promise<void> {
  await db.files.update(fileId, {
    data,
    updatedAt: Date.now(),
  });
}

export async function createFile(
  projectName: string,
  path: string,
  data: Uint8Array = new Uint8Array(0)
): Promise<ProjectFile> {
  const now = Date.now();
  const file: ProjectFile = {
    id: crypto.randomUUID(),
    projectName,
    path,
    data,
    createdAt: now,
    updatedAt: now,
  };

  await db.files.add(file);
  return file;
}

export async function deleteFile(fileId: string): Promise<void> {
  await db.files.delete(fileId);
}

export async function renameFile(
  fileId: string,
  newPath: string
): Promise<void> {
  await db.files.update(fileId, {
    path: newPath,
    updatedAt: Date.now(),
  });
}

// Utility Operations

export async function getProjectFileCount(
  projectName: string
): Promise<number> {
  return await db.files.where("projectName").equals(projectName).count();
}

export async function getProjectLastModified(
  projectName: string
): Promise<number> {
  const files = await db.files
    .where("projectName")
    .equals(projectName)
    .toArray();
  if (files.length === 0) return 0;
  return Math.max(...files.map(f => f.updatedAt));
}
