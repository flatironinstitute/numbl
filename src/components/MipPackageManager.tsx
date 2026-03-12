import { useEffect, useState, useCallback } from "react";
import {
  Box,
  Typography,
  IconButton,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Tooltip,
  CircularProgress,
} from "@mui/material";
import DeleteIcon from "@mui/icons-material/Delete";
import RefreshIcon from "@mui/icons-material/Refresh";
import { db, type MipPackageCache } from "../db/schema.js";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function estimateSize(pkg: MipPackageCache): number {
  let total = 0;
  for (const f of pkg.files) {
    total += f.source.length * 2; // rough UTF-16 estimate
    if (f.data) total += f.data.byteLength;
  }
  return total;
}

function formatDate(ts: number): string {
  return new Date(ts).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function MipPackageManager() {
  const [packages, setPackages] = useState<MipPackageCache[]>([]);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const all = await db.mipPackages.toArray();
      all.sort((a, b) => a.name.localeCompare(b.name));
      setPackages(all);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const deletePackage = async (name: string) => {
    await db.mipPackages.delete(name);
    setPackages(prev => prev.filter(p => p.name !== name));
  };

  const deleteAll = async () => {
    await db.mipPackages.clear();
    setPackages([]);
  };

  if (loading) {
    return (
      <Box sx={{ display: "flex", justifyContent: "center", pt: 4 }}>
        <CircularProgress size={24} />
      </Box>
    );
  }

  return (
    <Box sx={{ height: "100%", overflow: "auto", p: 1 }}>
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          mb: 1,
        }}
      >
        <Typography variant="body2" color="text.secondary">
          {packages.length} cached package{packages.length !== 1 ? "s" : ""}
        </Typography>
        <Box sx={{ display: "flex", gap: 0.5 }}>
          <Tooltip title="Refresh">
            <IconButton size="small" onClick={refresh}>
              <RefreshIcon fontSize="small" />
            </IconButton>
          </Tooltip>
          {packages.length > 0 && (
            <Button
              size="small"
              color="error"
              onClick={deleteAll}
              sx={{ fontSize: "0.75rem", textTransform: "none" }}
            >
              Delete All
            </Button>
          )}
        </Box>
      </Box>

      {packages.length === 0 ? (
        <Typography variant="body2" color="text.secondary">
          No MIP packages cached. Packages are downloaded when a script uses{" "}
          <code>mip load</code>.
        </Typography>
      ) : (
        <TableContainer>
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell
                  sx={{ fontWeight: 600, py: 0.5, fontSize: "0.8rem" }}
                >
                  Package
                </TableCell>
                <TableCell
                  sx={{ fontWeight: 600, py: 0.5, fontSize: "0.8rem" }}
                >
                  Version
                </TableCell>
                <TableCell
                  sx={{ fontWeight: 600, py: 0.5, fontSize: "0.8rem" }}
                  align="right"
                >
                  Files
                </TableCell>
                <TableCell
                  sx={{ fontWeight: 600, py: 0.5, fontSize: "0.8rem" }}
                  align="right"
                >
                  Size
                </TableCell>
                <TableCell
                  sx={{ fontWeight: 600, py: 0.5, fontSize: "0.8rem" }}
                >
                  Cached
                </TableCell>
                <TableCell sx={{ py: 0.5, width: 40 }} />
              </TableRow>
            </TableHead>
            <TableBody>
              {packages.map(pkg => (
                <TableRow key={pkg.name} hover>
                  <TableCell sx={{ py: 0.5, fontSize: "0.8rem" }}>
                    {pkg.name}
                  </TableCell>
                  <TableCell sx={{ py: 0.5, fontSize: "0.8rem" }}>
                    {pkg.version}
                  </TableCell>
                  <TableCell sx={{ py: 0.5, fontSize: "0.8rem" }} align="right">
                    {pkg.files.length}
                  </TableCell>
                  <TableCell sx={{ py: 0.5, fontSize: "0.8rem" }} align="right">
                    {formatBytes(estimateSize(pkg))}
                  </TableCell>
                  <TableCell sx={{ py: 0.5, fontSize: "0.8rem" }}>
                    {formatDate(pkg.fetchedAt)}
                  </TableCell>
                  <TableCell sx={{ py: 0.5 }}>
                    <Tooltip title={`Delete ${pkg.name}`}>
                      <IconButton
                        size="small"
                        color="error"
                        onClick={() => deletePackage(pkg.name)}
                      >
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    </Tooltip>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
      )}
    </Box>
  );
}
