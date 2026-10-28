import FullscreenIcon from "@mui/icons-material/Fullscreen";
import FullscreenExitIcon from "@mui/icons-material/FullscreenExit";
import SaveAltIcon from "@mui/icons-material/SaveAlt";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import {
  Alert,
  Box,
  IconButton,
  Link,
  Snackbar,
  Tab,
  Tabs,
  Tooltip,
  Typography,
} from "@mui/material";
import { useCallback, useEffect, useState } from "react";
import type { FiguresState } from "../../numbl-core/plotting/figuresReducer.js";
import { FigureView } from "../../graphics/FigureView.js";
import { downloadFigureHdf5 } from "../../graphics/exportFigureHdf5.js";
import { buildFigureViewerLink } from "../../graphics/openInFigureViewer.js";
import { uploadFigureForViewer } from "../../graphics/figureUpload.js";
import { compactTabSx } from "./panelStyles";
import { sortedHandles } from "./figureHandles";

/**
 * Desktop figures panel: one tab per figure, with open-in-viewer, HDF5
 * download, and an expand-to-full-window toggle (Escape collapses).
 */
export function FiguresPanel({
  figures,
  figureTab,
  onFigureTabChange,
}: {
  figures: FiguresState;
  figureTab: number;
  onFigureTabChange: (tab: number) => void;
}) {
  const handles = sortedHandles(figures.figs);
  const handle = handles[figureTab];
  const fig = figures.figs[handle];
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setExpanded(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded]);

  const [downloading, setDownloading] = useState(false);
  const handleDownload = useCallback(async () => {
    if (!fig) return;
    setDownloading(true);
    try {
      await downloadFigureHdf5(fig, handle);
    } catch (e) {
      console.error("Figure HDF5 export failed:", e);
    } finally {
      setDownloading(false);
    }
  }, [fig, handle]);

  // Hand-off banner state for figures too large to open via a direct link.
  const [viewerHandoff, setViewerHandoff] = useState<
    | { kind: "uploading" }
    | { kind: "ready"; url: string }
    | { kind: "failed"; baseUrl: string }
    | null
  >(null);
  const handleOpenInViewer = useCallback(async () => {
    if (!fig) return;
    const link = buildFigureViewerLink(fig);
    if (link.url) {
      window.open(link.url, "_blank");
      setViewerHandoff(null);
      return;
    }
    setViewerHandoff({ kind: "uploading" });
    try {
      const url = await uploadFigureForViewer(fig);
      setViewerHandoff({ kind: "ready", url });
    } catch {
      setViewerHandoff({ kind: "failed", baseUrl: link.baseUrl });
    }
  }, [fig]);

  return (
    <>
      <Box
        sx={
          expanded
            ? {
                position: "fixed",
                inset: 0,
                zIndex: 1250,
                bgcolor: "background.paper",
                display: "flex",
                flexDirection: "column",
              }
            : { height: "100%", display: "flex", flexDirection: "column" }
        }
      >
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            borderBottom: 1,
            borderColor: "divider",
          }}
        >
          <Tabs
            value={figureTab}
            onChange={(_, v) => onFigureTabChange(v)}
            variant="scrollable"
            scrollButtons="auto"
            sx={{ flexGrow: 1, ...compactTabSx }}
          >
            {handles.map(h => (
              <Tab key={h} label={`Figure ${h}`} />
            ))}
          </Tabs>
          <Tooltip title="Open this figure in the numbl figure viewer (new tab)">
            <span>
              <IconButton
                size="small"
                onClick={handleOpenInViewer}
                disabled={handles.length === 0}
                sx={{ mx: 0.5, flexShrink: 0 }}
              >
                <OpenInNewIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
          <Tooltip title="Download this figure's data as HDF5 (.h5)">
            <span>
              <IconButton
                size="small"
                onClick={handleDownload}
                disabled={downloading || handles.length === 0}
                sx={{ mx: 0.5, flexShrink: 0 }}
              >
                <SaveAltIcon fontSize="small" />
              </IconButton>
            </span>
          </Tooltip>
          <Tooltip
            title={
              expanded
                ? "Collapse figures (Esc)"
                : "Expand figures to full window"
            }
          >
            <IconButton
              size="small"
              onClick={() => setExpanded(v => !v)}
              sx={{ mx: 0.5, flexShrink: 0 }}
            >
              {expanded ? (
                <FullscreenExitIcon fontSize="small" />
              ) : (
                <FullscreenIcon fontSize="small" />
              )}
            </IconButton>
          </Tooltip>
        </Box>
        <Box sx={{ flexGrow: 1, overflow: "auto", p: 1 }}>
          {handles.length > 0 ? (
            <FigureView figure={fig} />
          ) : (
            <Typography
              variant="body2"
              color="text.disabled"
              sx={{ p: 2, textAlign: "center", fontStyle: "italic" }}
            >
              No figures
            </Typography>
          )}
        </Box>
      </Box>
      <Snackbar
        open={viewerHandoff !== null}
        onClose={() => setViewerHandoff(null)}
        anchorOrigin={{ vertical: "bottom", horizontal: "center" }}
      >
        <Alert
          severity={viewerHandoff?.kind === "failed" ? "warning" : "info"}
          onClose={() => setViewerHandoff(null)}
          sx={{ width: "100%" }}
        >
          {viewerHandoff?.kind === "uploading" &&
            "Preparing a temporary link to the figure viewer…"}
          {viewerHandoff?.kind === "ready" && (
            <>
              Figure too large for a direct link — uploaded (encrypted,
              temporary).{" "}
              <Link href={viewerHandoff.url} target="_blank" rel="noreferrer">
                Open in figure viewer
              </Link>
            </>
          )}
          {viewerHandoff?.kind === "failed" && (
            <>
              This figure is too large to open through a link, and the temporary
              upload failed. Download it (
              <SaveAltIcon
                fontSize="inherit"
                sx={{ verticalAlign: "middle" }}
              />
              ) and{" "}
              <Link
                href={viewerHandoff.baseUrl}
                target="_blank"
                rel="noreferrer"
              >
                open the figure viewer
              </Link>
              , then load the file.
            </>
          )}
        </Alert>
      </Snackbar>
    </>
  );
}
