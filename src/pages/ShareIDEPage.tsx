import SaveIcon from "@mui/icons-material/Save";
import ShareIcon from "@mui/icons-material/Share";
import HomeIcon from "@mui/icons-material/Home";
import {
  Box,
  Button,
  CircularProgress,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  IconButton,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import { useCallback, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useShareProjectFiles } from "../hooks/useShareProjectFiles";
import { IDEWorkspace } from "../components/IDEWorkspace";
import {
  createProject,
  createFile,
  listProjects,
  getProjectFiles,
  deleteFile as deleteDbFile,
} from "../db/operations";
import { validateProjectName } from "../utils/validation";

export function ShareIDEPage() {
  const navigate = useNavigate();
  const {
    files,
    activeFileId,
    loading,
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
    urlSizeTooLarge,
  } = useShareProjectFiles();

  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [projectName, setProjectName] = useState("");
  const [saveError, setSaveError] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [copyFeedback, setCopyFeedback] = useState(false);

  const handleCopyUrl = useCallback(() => {
    navigator.clipboard.writeText(window.location.href).then(() => {
      setCopyFeedback(true);
      setTimeout(() => setCopyFeedback(false), 2000);
    });
  }, []);

  const handleSaveAsProject = useCallback(async () => {
    if (!projectName.trim()) {
      setSaveError("Project name is required");
      return;
    }

    try {
      setIsSaving(true);
      setSaveError("");

      const existingProjects = await listProjects();
      const existingNames = existingProjects.map(p => p.name);
      const validation = validateProjectName(projectName, existingNames);
      if (!validation.valid) {
        setSaveError(validation.error || "Invalid project name");
        setIsSaving(false);
        return;
      }

      await createProject(projectName);

      // createProject creates a default script.m - we need to replace all files
      const defaultFiles = await getProjectFiles(projectName);
      for (const f of defaultFiles) {
        await deleteDbFile(f.id);
      }

      // Create all the shared files in the new project
      for (const file of files) {
        await createFile(projectName, file.name, file.data);
      }

      setSaveDialogOpen(false);
      navigate(`/project/${projectName}`);
    } catch (e) {
      console.error("Failed to save project:", e);
      setSaveError(e instanceof Error ? e.message : "Failed to save project");
    } finally {
      setIsSaving(false);
    }
  }, [projectName, files, navigate]);

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
      <IconButton size="small" onClick={() => navigate("/")} sx={{ mr: 0.5 }}>
        <HomeIcon fontSize="small" />
      </IconButton>
      <Typography
        variant="body2"
        fontWeight="medium"
        sx={{ flexGrow: 1 }}
        color="text.secondary"
      >
        Shared workspace
      </Typography>
      {urlSizeTooLarge && (
        <Typography
          variant="caption"
          color="warning.main"
          sx={{ mr: 1, fontSize: "0.7rem" }}
        >
          URL too large to share
        </Typography>
      )}
      <Tooltip title={copyFeedback ? "Copied!" : "Copy URL"}>
        <IconButton size="small" onClick={handleCopyUrl} sx={{ mr: 0.5 }}>
          <ShareIcon fontSize="small" />
        </IconButton>
      </Tooltip>
      <Button
        variant="outlined"
        size="small"
        startIcon={<SaveIcon />}
        onClick={() => {
          setProjectName("");
          setSaveError("");
          setSaveDialogOpen(true);
        }}
        sx={{ fontSize: "0.75rem", py: 0.25 }}
      >
        Save as project
      </Button>
    </>
  );

  return (
    <>
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
      />

      <Dialog
        open={saveDialogOpen}
        onClose={() => setSaveDialogOpen(false)}
        maxWidth="xs"
        fullWidth
      >
        <DialogTitle>Save as Project</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            label="Project name"
            value={projectName}
            onChange={e => {
              setProjectName(e.target.value);
              setSaveError("");
            }}
            onKeyDown={e => {
              if (e.key === "Enter") handleSaveAsProject();
            }}
            size="small"
            fullWidth
            error={!!saveError}
            helperText={
              saveError || "Letters, numbers, dashes, and underscores only"
            }
            sx={{ mt: 1 }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSaveDialogOpen(false)}>Cancel</Button>
          <Button
            variant="contained"
            onClick={handleSaveAsProject}
            disabled={isSaving || !projectName.trim()}
          >
            {isSaving ? "Saving..." : "Save"}
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}
