import ShareIcon from "@mui/icons-material/Share";
import GitHubIcon from "@mui/icons-material/GitHub";
import {
  Box,
  CircularProgress,
  IconButton,
  Link,
  Tooltip,
  Typography,
} from "@mui/material";
import { useCallback, useState } from "react";
import { useStaticProjectFiles } from "../hooks/useStaticProjectFiles";
import { IDEWorkspace } from "../components/IDEWorkspace";
import { encodeShareData } from "../utils/shareUrl";

// Shared links always point at the canonical numbl IDE, not the host this
// static bundle happens to be deployed on (GitHub Pages, custom domain, etc.).
const NUMBL_ORIGIN = "https://numbl.org";

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
    repository,
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

  const [copyFeedback, setCopyFeedback] = useState(false);

  const handleShare = useCallback(async () => {
    try {
      const allContents = await loadAllContents();
      const encoded = encodeShareData(files, allContents, activeFileId);
      const url = `${NUMBL_ORIGIN}/share#${encoded}`;
      if (url.length > 64000) {
        alert(
          "Project is too large to share via URL. Try reducing the number or size of files."
        );
        return;
      }
      await navigator.clipboard.writeText(url);
      setCopyFeedback(true);
      setTimeout(() => setCopyFeedback(false), 2000);
    } catch (e) {
      console.error("Failed to generate share URL:", e);
      alert("Failed to generate share URL.");
    }
  }, [files, activeFileId, loadAllContents]);

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
      {repository && (
        <Tooltip title="View source repository">
          <IconButton
            size="small"
            component="a"
            href={repository}
            target="_blank"
            rel="noopener noreferrer"
            sx={{ mr: 0.5 }}
          >
            <GitHubIcon fontSize="small" />
          </IconButton>
        </Tooltip>
      )}
      <Tooltip title={copyFeedback ? "Copied!" : "Copy shareable URL"}>
        <IconButton size="small" onClick={handleShare} sx={{ mr: 0.5 }}>
          <ShareIcon fontSize="small" />
        </IconButton>
      </Tooltip>
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
