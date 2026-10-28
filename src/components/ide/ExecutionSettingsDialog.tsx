import ComputerIcon from "@mui/icons-material/Computer";
import DnsIcon from "@mui/icons-material/Dns";
import {
  Box,
  Button,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  FormControlLabel,
  Switch,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from "@mui/material";
import { useState } from "react";
import type { RemoteExecution } from "../../hooks/useRemoteExecution";
import {
  effectiveWasmBridgeUrl,
  getWasmBridgeUrl,
  isWasmBridgeEnabled,
  setWasmBridgeEnabled,
  setWasmBridgeUrl,
} from "../../utils/wasmLapackBridge";

const monoInput = {
  input: { style: { fontFamily: "monospace", fontSize: "0.85rem" } },
};

/**
 * Where scripts run (browser or a local `numbl serve`) and the WASM matrix
 * accelerator. The server URL and accelerator settings are drafts until
 * Save; the browser/server switch applies immediately.
 */
export function ExecutionSettingsDialog({
  open,
  onClose,
  remote,
  onWasmBridgeChange,
}: {
  open: boolean;
  onClose: () => void;
  remote: RemoteExecution;
  /** Called on Save with the effective accelerator URL (null = off). */
  onWasmBridgeChange: (url: string | null) => void;
}) {
  const [urlDraft, setUrlDraft] = useState(remote.serviceUrl);
  const [wasmEnabledDraft, setWasmEnabledDraft] = useState(isWasmBridgeEnabled);
  const [wasmUrlDraft, setWasmUrlDraft] = useState(getWasmBridgeUrl);

  // Start the URL draft from the saved URL each time the dialog opens.
  const [wasOpen, setWasOpen] = useState(open);
  if (open !== wasOpen) {
    setWasOpen(open);
    if (open) setUrlDraft(remote.serviceUrl);
  }

  const handleSave = () => {
    remote.setServiceUrl(urlDraft);
    // Persist WASM accelerator settings and push the new effective URL to
    // the worker (null ⇒ uninstall) so it takes effect without a reload.
    setWasmBridgeEnabled(wasmEnabledDraft);
    setWasmBridgeUrl(wasmUrlDraft);
    onWasmBridgeChange(effectiveWasmBridgeUrl());
    onClose();
  };

  const { enabled, serverStatus, nativeAddon, passkey } = remote;
  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Execution Settings</DialogTitle>
      <DialogContent
        sx={{
          display: "flex",
          flexDirection: "column",
          gap: 2,
          pt: 3,
          overflow: "visible",
        }}
      >
        <ToggleButtonGroup
          value={enabled ? "local" : "browser"}
          exclusive
          onChange={(_, mode: "browser" | "local" | null) => {
            if (mode !== null) remote.setEnabled(mode === "local");
          }}
          size="small"
        >
          <ToggleButton value="browser" sx={{ px: 2 }}>
            <ComputerIcon sx={{ fontSize: "1rem", mr: 0.5 }} />
            In browser
          </ToggleButton>
          <ToggleButton value="local" sx={{ px: 2 }}>
            <DnsIcon sx={{ fontSize: "1rem", mr: 0.5 }} />
            Local server
          </ToggleButton>
        </ToggleButtonGroup>
        <TextField
          label="Server URL"
          value={urlDraft}
          onChange={e => setUrlDraft(e.target.value)}
          size="small"
          fullWidth
          disabled={!enabled}
          slotProps={monoInput}
          helperText="Default: http://localhost:3001"
        />
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <Typography
            variant="body2"
            color={
              !enabled
                ? "text.disabled"
                : serverStatus === "connected"
                  ? "success.main"
                  : serverStatus === "disconnected"
                    ? "warning.main"
                    : "text.secondary"
            }
          >
            {!enabled
              ? "Server status: n/a"
              : serverStatus === "connected"
                ? "Connected" +
                  (nativeAddon === true
                    ? " (native addon available)"
                    : nativeAddon === false
                      ? " (no native addon)"
                      : "")
                : serverStatus === "disconnected"
                  ? "Not connected"
                  : "Checking..."}
          </Typography>
          <Button
            size="small"
            disabled={!enabled}
            onClick={() => remote.checkHealth(urlDraft)}
            sx={{ minWidth: 0, textTransform: "none" }}
          >
            Check
          </Button>
        </Box>
        <Box sx={{ opacity: enabled ? 1 : 0.4 }}>
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Start a local server in a terminal:
          </Typography>
          <Box
            component="pre"
            sx={{
              bgcolor: "#1e1e1e",
              color: "#ccc",
              p: 1.5,
              borderRadius: 1,
              fontSize: "0.8rem",
              m: 0,
              whiteSpace: "pre-wrap",
            }}
          >
            {`# If installed globally:\nnpx numbl serve --passkey ${passkey}\n\n# Or from the repo (dev mode):\nnpx tsx src/cli.ts serve --passkey ${passkey}`}
          </Box>
          <Typography
            variant="caption"
            color="text.disabled"
            onClick={remote.regeneratePasskey}
            sx={{
              cursor: "pointer",
              mt: 0.5,
              "&:hover": { color: "text.secondary" },
            }}
          >
            regenerate passkey
          </Typography>
        </Box>
        <Divider />
        <Box>
          <FormControlLabel
            control={
              <Switch
                checked={wasmEnabledDraft}
                onChange={e => setWasmEnabledDraft(e.target.checked)}
                size="small"
              />
            }
            label="Accelerate matrix math (WASM)"
          />
          <Typography variant="caption" color="text.secondary" component="p">
            Loads an external WebAssembly LAPACK/BLAS bridge to speed up matrix
            multiplication in the browser. Falls back to the built-in pure-JS
            implementation when off or unavailable. Turning it off is temporary
            — it re-enables on page reload.
          </Typography>
          <TextField
            label="Bridge URL"
            value={wasmUrlDraft}
            onChange={e => setWasmUrlDraft(e.target.value)}
            size="small"
            fullWidth
            disabled={!wasmEnabledDraft}
            sx={{ mt: 1 }}
            slotProps={monoInput}
            helperText="Base URL serving numbl-bridge.json"
          />
        </Box>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button variant="contained" onClick={handleSave}>
          Save
        </Button>
      </DialogActions>
    </Dialog>
  );
}
