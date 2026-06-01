import { Box, CircularProgress, Link, Typography } from "@mui/material";
import { useStaticProjectFiles } from "../hooks/useStaticProjectFiles";
import { IDEWorkspace } from "../components/IDEWorkspace";

/**
 * Standalone app for a statically-deployed numbl project. Loads the baked-in
 * project bundle and renders the full IDE workspace against it. Files are
 * editable and runnable in-memory; the deployed bundle is the source of truth.
 */
export function SiteApp() {
  const {
    files,
    activeFileId,
    loading,
    title,
    loadError,
    setActiveFileId,
    updateFileContent,
    addFile,
    addFolder,
    deleteFile,
    deleteFolder,
    renameFile,
    renameFolder,
    moveFile,
    duplicateFile,
    uploadFiles,
    loadFileContent,
    loadAllContents,
    contentCache,
  } = useStaticProjectFiles();

  if (loading) {
    return (
      <Box
        sx={{
          display: "flex",
          justifyContent: "center",
          alignItems: "center",
          height: "100vh",
        }}
      >
        <CircularProgress />
      </Box>
    );
  }

  const headerContent = (
    <>
      <Typography
        variant="body2"
        fontWeight="medium"
        sx={{ flexGrow: 1 }}
        noWrap
      >
        {title || "numbl project"}
        {loadError && (
          <Typography
            component="span"
            variant="caption"
            color="warning.main"
            sx={{ ml: 1 }}
          >
            (bundle failed to load)
          </Typography>
        )}
      </Typography>
      <Link
        href="https://numbl.org"
        target="_blank"
        rel="noopener noreferrer"
        variant="caption"
        underline="hover"
        sx={{ opacity: 0.7, "&:hover": { opacity: 1 } }}
      >
        powered by numbl
      </Link>
    </>
  );

  return (
    <IDEWorkspace
      files={files}
      activeFileId={activeFileId}
      setActiveFileId={setActiveFileId}
      updateFileContent={updateFileContent}
      addFile={addFile}
      addFolder={addFolder}
      deleteFile={deleteFile}
      deleteFolder={deleteFolder}
      renameFile={renameFile}
      renameFolder={renameFolder}
      moveFile={moveFile}
      duplicateFile={duplicateFile}
      uploadFiles={uploadFiles}
      headerContent={headerContent}
      loadFileContent={loadFileContent}
      loadAllContents={loadAllContents}
      contentCache={contentCache}
    />
  );
}
