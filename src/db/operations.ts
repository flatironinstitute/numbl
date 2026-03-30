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

  await db.transaction(
    "rw",
    db.projects,
    db.files,
    db.fileContents,
    async () => {
      await db.projects.add({
        name,
        createdAt: now,
        updatedAt: now,
        lastOpenedAt: now,
      });

      const id = crypto.randomUUID();
      const data = textEncoder.encode(DEFAULT_CODE);
      await db.files.add({
        id,
        projectName: name,
        path: "script.m",
        createdAt: now,
        updatedAt: now,
      });
      await db.fileContents.add({ id, data });
    }
  );
}

export async function getProject(name: string): Promise<Project | undefined> {
  return await db.projects.get(name);
}

export async function listProjects(): Promise<Project[]> {
  const projects = await db.projects
    .orderBy("lastOpenedAt")
    .reverse()
    .toArray();
  return projects.filter(p => p.name !== "__home__");
}

export async function deleteProject(name: string): Promise<void> {
  await db.transaction(
    "rw",
    db.projects,
    db.files,
    db.fileContents,
    async () => {
      const fileIds = await db.files
        .where("projectName")
        .equals(name)
        .primaryKeys();
      await db.fileContents.bulkDelete(fileIds);
      await db.files.where("projectName").equals(name).delete();
      await db.projects.delete(name);
    }
  );
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

    await db.files
      .where("projectName")
      .equals(oldName)
      .modify({ projectName: newName });

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

// File Operations — metadata only

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

// File content operations

export async function getFileContent(fileId: string): Promise<Uint8Array> {
  const record = await db.fileContents.get(fileId);
  return record?.data ?? new Uint8Array(0);
}

export async function getAllFileContents(
  projectName: string
): Promise<Map<string, Uint8Array>> {
  const fileIds = await db.files
    .where("projectName")
    .equals(projectName)
    .primaryKeys();
  const contents = await db.fileContents.bulkGet(fileIds);
  const map = new Map<string, Uint8Array>();
  for (let i = 0; i < fileIds.length; i++) {
    map.set(fileIds[i], contents[i]?.data ?? new Uint8Array(0));
  }
  return map;
}

export async function saveFileData(
  fileId: string,
  data: Uint8Array
): Promise<void> {
  await db.transaction("rw", db.files, db.fileContents, async () => {
    await db.fileContents.put({ id: fileId, data });
    await db.files.update(fileId, { updatedAt: Date.now() });
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
    createdAt: now,
    updatedAt: now,
  };

  await db.transaction("rw", db.files, db.fileContents, async () => {
    await db.files.add(file);
    await db.fileContents.add({ id: file.id, data });
  });
  return file;
}

export async function deleteFile(fileId: string): Promise<void> {
  await db.transaction("rw", db.files, db.fileContents, async () => {
    await db.files.delete(fileId);
    await db.fileContents.delete(fileId);
  });
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

// Home directory operations (shared across all projects)

const HOME_PROJECT_NAME = "__home__";

export async function getHomeFiles(): Promise<ProjectFile[]> {
  return await db.files
    .where("projectName")
    .equals(HOME_PROJECT_NAME)
    .toArray();
}

export async function getHomeFileContents(): Promise<Map<string, Uint8Array>> {
  return getAllFileContents(HOME_PROJECT_NAME);
}

export async function ensureHomeProject(): Promise<void> {
  const existing = await db.projects.get(HOME_PROJECT_NAME);
  if (!existing) {
    const now = Date.now();
    await db.projects.add({
      name: HOME_PROJECT_NAME,
      createdAt: now,
      updatedAt: now,
      lastOpenedAt: now,
    });
  }
}
