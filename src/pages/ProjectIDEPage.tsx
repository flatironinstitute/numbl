import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import ShareIcon from "@mui/icons-material/Share";
import {
  Box,
  Button,
  CircularProgress,
  IconButton,
  Tooltip,
  Typography,
} from "@mui/material";
import { useCallback, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useProject } from "../hooks/useProject";
import { useProjectFiles } from "../hooks/useProjectFiles";
import { IDEWorkspace } from "../components/IDEWorkspace";
import { encodeShareData } from "../utils/shareUrl";

export function ProjectIDEPage() {
  const { projectName } = useParams<{ projectName: string }>();
  const navigate = useNavigate();
  const [copyFeedback, setCopyFeedback] = useState(false);

  const {
    project,
    loading: projectLoading,
    error: projectError,
  } = useProject(projectName!);
  const {
    files,
    activeFileId,
    loading: filesLoading,
    setActiveFileId,
    updateFileContent,
    addFile,
    addFolder,
    deleteFile,
    deleteFolder,
    renameFile,
    renameFolder,
    moveFile,
    uploadFiles,
    loadFileContent,
    loadAllContents,
    contentCache,
    mergeVfsChanges,
  } = useProjectFiles(projectName!);

  const handleShare = useCallback(async () => {
    try {
      const allContents = await loadAllContents();
      const encoded = encodeShareData(files, allContents, activeFileId);
      const url = `${window.location.origin}/share#${encoded}`;
      if (url.length > 64000) {
        alert(
          "Project is too large to share via URL. Try reducing the number or size of files."
        );
        return;
      }
      navigator.clipboard.writeText(url).then(() => {
        setCopyFeedback(true);
        setTimeout(() => setCopyFeedback(false), 2000);
      });
    } catch (e) {
      console.error("Failed to generate share URL:", e);
      alert("Failed to generate share URL.");
    }
  }, [files, activeFileId, loadAllContents]);

  if (projectLoading || filesLoading) {
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

  if (projectError) {
    return (
      <Box sx={{ p: 4 }}>
        <Typography variant="h4" color="error" gutterBottom>
          Error Loading Project
        </Typography>
        <Typography variant="body1">{projectError.message}</Typography>
        <Button
          variant="contained"
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate("/")}
          sx={{ mt: 2 }}
        >
          Back to Projects
        </Button>
      </Box>
    );
  }

  if (!project) {
    return (
      <Box sx={{ p: 4 }}>
        <Typography variant="h4" gutterBottom>
          Project not found
        </Typography>
        <Button
          variant="contained"
          startIcon={<ArrowBackIcon />}
          onClick={() => navigate("/")}
          sx={{ mt: 2 }}
        >
          Back to Projects
        </Button>
      </Box>
    );
  }

  const headerContent = (
    <>
      <IconButton size="small" onClick={() => navigate("/")} sx={{ mr: 0.5 }}>
        <ArrowBackIcon fontSize="small" />
      </IconButton>
      <Typography variant="body2" fontWeight="medium" sx={{ flexGrow: 1 }}>
        {project.displayName || project.name}
      </Typography>
      <Tooltip title={copyFeedback ? "Copied!" : "Copy shareable URL"}>
        <IconButton size="small" onClick={handleShare}>
          <ShareIcon fontSize="small" />
        </IconButton>
      </Tooltip>
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
      uploadFiles={uploadFiles}
      headerContent={headerContent}
      projectName={projectName}
      loadFileContent={loadFileContent}
      loadAllContents={loadAllContents}
      contentCache={contentCache}
      mergeVfsChanges={mergeVfsChanges}
    />
  );
}
