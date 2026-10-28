import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import StopIcon from "@mui/icons-material/Stop";
import ComputerIcon from "@mui/icons-material/Computer";
import DnsIcon from "@mui/icons-material/Dns";
import NoteAddIcon from "@mui/icons-material/NoteAdd";
import {
  Box,
  Button,
  IconButton,
  Tab,
  Tabs,
  ToggleButton,
  ToggleButtonGroup,
  Tooltip,
  Typography,
} from "@mui/material";
import type { ReactNode } from "react";
import type { OptLevel } from "../../numbl-core/executors/plugins.js";
import type { FileEntry } from "../../hooks/projectFilesState";
import { fileText, isBinaryData } from "../../hooks/useProjectFiles";
import type { RemoteExecution } from "../../hooks/useRemoteExecution";
import { MarkdownView } from "../MarkdownView";
import { FileEditor } from "./FileEditor";
import { panelTabsSx } from "./panelStyles";

/** Script / REPL tabs. The REPL tab is hidden when `showRepl` is false. */
export function EditorPanel({
  showRepl,
  tab,
  onTabChange,
  script,
  repl,
}: {
  showRepl: boolean;
  tab: number;
  onTabChange: (tab: number) => void;
  script: ReactNode;
  repl: ReactNode;
}) {
  const current = showRepl ? tab : 0;
  return (
    <Box sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <Tabs
        value={current}
        onChange={(_, v) => onTabChange(v)}
        sx={panelTabsSx}
      >
        <Tab label="Script" />
        {showRepl && <Tab label="REPL" />}
      </Tabs>
      <Box sx={{ flexGrow: 1, overflow: "hidden" }}>
        {current === 0 ? script : repl}
      </Box>
    </Box>
  );
}

const toggleChipSx = (on: boolean) => ({
  cursor: "pointer",
  fontSize: "0.7rem",
  px: 0.5,
  py: 0.1,
  borderRadius: 0.5,
  bgcolor: on ? "action.selected" : "transparent",
  opacity: on ? 1 : 0.5,
  "&:hover": { opacity: 1 },
  userSelect: "none",
});

/** Run/Stop button, execution target, and the persist / JIT toggles. */
export function RunControls({
  isRunning,
  onRun,
  onStop,
  remote,
  onOpenSettings,
  persistWorkspace,
  onTogglePersist,
  optimization,
  onToggleOptimization,
}: {
  isRunning: boolean;
  onRun: () => void;
  onStop: () => void;
  remote: RemoteExecution;
  onOpenSettings: () => void;
  persistWorkspace: boolean;
  onTogglePersist: () => void;
  optimization: OptLevel;
  onToggleOptimization: () => void;
}) {
  const { enabled: isRemote, serverStatus, nativeAddon } = remote;
  return (
    <>
      <Button
        variant="contained"
        color={isRunning ? "error" : "success"}
        size="small"
        startIcon={isRunning ? <StopIcon /> : <PlayArrowIcon />}
        onClick={isRunning ? onStop : onRun}
        sx={{
          py: 0.25,
          px: 2,
          fontSize: "0.8rem",
          fontWeight: 600,
          textTransform: "none",
          borderRadius: 1.5,
          boxShadow: "none",
          "&:hover": { boxShadow: "none" },
        }}
      >
        {isRunning ? "Stop" : "Run"}
      </Button>
      <Tooltip
        title={
          !isRemote
            ? "Executing in browser"
            : "Executing on local server" +
              (serverStatus === "disconnected" ? " (not connected)" : "") +
              (nativeAddon === true
                ? " (native)"
                : nativeAddon === false
                  ? " (no native addon)"
                  : "")
        }
      >
        <IconButton
          size="small"
          onClick={onOpenSettings}
          sx={{ opacity: 0.5, "&:hover": { opacity: 1 } }}
        >
          {!isRemote ? (
            <ComputerIcon sx={{ fontSize: "0.9rem" }} />
          ) : (
            <DnsIcon sx={{ fontSize: "0.9rem", color: "error.main" }} />
          )}
        </IconButton>
      </Tooltip>
      {isRemote && (
        <Typography
          variant="caption"
          sx={{ fontSize: "0.65rem", opacity: 0.5, color: "error.main" }}
        >
          {serverStatus === "disconnected"
            ? "no server"
            : nativeAddon
              ? "native"
              : nativeAddon === false
                ? "no native"
                : "local"}
        </Typography>
      )}
      {!isRemote && (
        <Tooltip
          title={
            persistWorkspace
              ? "Workspace persists across runs and REPL (click for fresh each run)"
              : "Fresh workspace each run (click to persist across runs and REPL)"
          }
        >
          <Typography
            variant="caption"
            onClick={onTogglePersist}
            sx={toggleChipSx(persistWorkspace)}
          >
            {persistWorkspace ? "persist" : "1x"}
          </Typography>
        </Tooltip>
      )}
      <Tooltip
        title={
          optimization === "0"
            ? "Interpreter only (click for JIT)"
            : "JIT (click to disable)"
        }
      >
        <Typography
          variant="caption"
          onClick={onToggleOptimization}
          sx={toggleChipSx(optimization !== "0")}
        >
          {optimization === "0" ? "no jit" : "jit"}
        </Typography>
      </Tooltip>
    </>
  );
}

/**
 * The Script tab: a toolbar (run controls for .m files, a rendered/source
 * toggle for Markdown, the file name) over the file's editor or viewer.
 */
export function ScriptPane({
  file,
  content,
  runControls,
  mdViewMode,
  onMdViewModeChange,
  onChange,
  onMarkdownFileLink,
  onAddFile,
}: {
  file: FileEntry | undefined;
  /** The file's content, or null while it loads. */
  content: Uint8Array | null;
  /** Shown for runnable (.m) files. */
  runControls: ReactNode;
  mdViewMode: "rendered" | "source";
  onMdViewModeChange: (mode: "rendered" | "source") => void;
  onChange: (value: string) => void;
  onMarkdownFileLink: (href: string) => void;
  onAddFile: () => void;
}) {
  if (!file) {
    return (
      <Box
        sx={{
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexDirection: "column",
          gap: 2,
          p: 3,
        }}
      >
        <Typography variant="body2" color="text.secondary">
          No file open
        </Typography>
        <Button
          variant="contained"
          size="small"
          startIcon={<NoteAddIcon />}
          onClick={onAddFile}
          sx={{ textTransform: "none" }}
        >
          New file
        </Button>
      </Box>
    );
  }

  const lower = file.name.toLowerCase();
  const isMarkdown = lower.endsWith(".md");
  const isRunnable = lower.endsWith(".m");
  const isBinary = content ? isBinaryData(content) : false;

  return (
    <Box sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <Box
        sx={{
          px: 1.5,
          py: 0.5,
          borderBottom: 1,
          borderColor: "divider",
          display: "flex",
          alignItems: "center",
          gap: 1,
          bgcolor: "background.default",
        }}
      >
        {isRunnable && runControls}
        {isMarkdown && (
          <ToggleButtonGroup
            value={mdViewMode}
            exclusive
            size="small"
            onChange={(_, v) => {
              if (v) onMdViewModeChange(v);
            }}
            sx={{
              "& .MuiToggleButton-root": {
                py: 0,
                px: 1,
                fontSize: "0.7rem",
                textTransform: "none",
                lineHeight: 1.6,
              },
            }}
          >
            <ToggleButton value="rendered">Rendered</ToggleButton>
            <ToggleButton value="source">Source</ToggleButton>
          </ToggleButtonGroup>
        )}
        <Typography
          variant="caption"
          color="text.secondary"
          sx={{ fontFamily: "monospace", fontWeight: "bold" }}
        >
          {file.name}
        </Typography>
      </Box>

      <Box sx={{ flexGrow: 1, minHeight: 0 }}>
        {isBinary ? (
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              height: "100%",
              color: "text.secondary",
            }}
          >
            <Typography variant="body2">
              Binary file ({content?.length ?? 0} bytes)
            </Typography>
          </Box>
        ) : isMarkdown && mdViewMode === "rendered" ? (
          <Box sx={{ height: "100%", overflow: "auto", p: 3 }}>
            <MarkdownView
              source={content ? fileText(content) : ""}
              onFileLink={onMarkdownFileLink}
            />
          </Box>
        ) : (
          <FileEditor
            fileId={file.id}
            fileName={file.name}
            content={content}
            onChange={onChange}
          />
        )}
      </Box>
    </Box>
  );
}
