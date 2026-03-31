import { useState, useEffect, useMemo, useCallback } from "react";
import {
  Box,
  Typography,
  Button,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  DialogContentText,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Tooltip,
  Chip,
  Collapse,
} from "@mui/material";
import ExpandMoreIcon from "@mui/icons-material/ExpandMore";
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined";
import AddIcon from "@mui/icons-material/Add";
import DeleteIcon from "@mui/icons-material/Delete";
import EditIcon from "@mui/icons-material/Edit";
import GitHubIcon from "@mui/icons-material/GitHub";
import TerminalIcon from "@mui/icons-material/Terminal";
import { useNavigate } from "react-router-dom";
import { CreateProjectDialog } from "../components/CreateProjectDialog";
import {
  listProjects,
  deleteProject,
  renameProject,
  getProjectFileCount,
  getProjectLastModified,
} from "../db/operations";
import { validateProjectName } from "../utils/validation";
import type { Project } from "../db/schema";
import { getAllIBuiltinNames } from "../numbl-core/interpreter/builtins/index.js";
import { getAllConstantNames } from "../numbl-core/helpers/constants.js";
import { getDummyBuiltinNames } from "../numbl-core/helpers/dummy.js";
import { SPECIAL_BUILTIN_NAMES } from "../numbl-core/runtime/specialBuiltinNames.js";

interface ProjectWithMetadata extends Project {
  fileCount: number;
  lastModified: number;
}

function formatDate(timestamp: number): string {
  if (!timestamp) return "Never";
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / 60000);
  const diffHours = Math.floor(diffMs / 3600000);
  const diffDays = Math.floor(diffMs / 86400000);

  if (diffMins < 1) return "Just now";
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;

  return date.toLocaleDateString();
}

const CAPABILITIES: { label: string; items: string[] }[] = [
  {
    label: "Language",
    items: [
      "classes & inheritance",
      "abstract classes",
      "handle classes",
      "enumerations",
      "namespaces & packages",
      "nested functions",
      "anonymous functions & closures",
      "function handles",
      "global & persistent variables",
      "try / catch / error / warning",
      "switch / case",
      "for / while loops",
      "varargin / varargout",
      "argument validation",
      "import statements",
      "regular expressions",
    ],
  },
  {
    label: "Data types",
    items: [
      "N-D arrays",
      "complex numbers",
      "logical arrays",
      "cell arrays",
      "structs & struct arrays",
      "sparse matrices",
      "dictionaries",
      "char arrays & strings",
      "function handles",
    ],
  },
  {
    label: "Operators",
    items: [
      "matrix arithmetic (* / \\ ^)",
      "element-wise (.* ./ .^)",
      "comparison (== ~= < > <= >=)",
      "logical (&& || & | ~)",
      "bitwise (bitand, bitor, bitxor, bitshift)",
      "transpose (' .')",
      "colon ranges (a:b, a:s:b)",
      "concatenation ([ ; ])",
    ],
  },
  {
    label: "Numerics",
    items: [
      "linear algebra (SVD, QR, QZ, LU, Cholesky, eig)",
      "FFT / IFFT",
      "interpolation (interp1)",
      "polynomials (polyfit, polyval, roots, conv)",
      "statistics (mean, std, var, cov, corrcoef)",
      "set operations (union, intersect, setdiff, unique)",
      "special functions (Bessel, Airy, erf, gamma, beta)",
      "random number generation (rand, randn, randi, rng)",
      "numerical integration (trapz, cumtrapz)",
    ],
  },
  {
    label: "Plotting",
    items: [
      "2-D (plot, scatter, imagesc)",
      "3-D (plot3, surf, mesh, waterfall)",
      "contour & contourf",
      "colormaps, colorbar, legend, subplot",
    ],
  },
  {
    label: "I/O & system",
    items: [
      "file I/O (fopen, fread, fwrite, fileread)",
      "JSON (jsondecode)",
      "web (webread, websave)",
      "path & directory operations",
      "environment variables",
      "sprintf / sscanf",
    ],
  },
  {
    label: "Engine",
    items: [
      "JIT compilation to JavaScript",
      "type inference & specialization",
      "copy-on-write arrays",
      "column-major storage",
    ],
  },
];

function CapabilitySection({
  label,
  items,
  mono,
}: {
  label: string;
  items: string[];
  mono?: boolean;
}) {
  return (
    <Box sx={{ mb: 2 }}>
      <Typography
        variant="caption"
        color="text.secondary"
        sx={{
          textTransform: "uppercase",
          letterSpacing: "0.08em",
          display: "block",
          mb: 0.75,
        }}
      >
        {label}
      </Typography>
      <Box
        sx={{
          display: "flex",
          flexWrap: "wrap",
          gap: 0.5,
          justifyContent: "center",
        }}
      >
        {items.map(item => (
          <Chip
            key={item}
            label={item}
            size="small"
            variant="outlined"
            sx={{
              fontSize: "0.7rem",
              height: 22,
              borderColor: "divider",
              color: "text.secondary",
              ...(mono && { fontFamily: "monospace" }),
            }}
          />
        ))}
      </Box>
    </Box>
  );
}

export function ProjectListPage() {
  const [projects, setProjects] = useState<ProjectWithMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [newName, setNewName] = useState("");
  const [renameError, setRenameError] = useState("");
  const [builtinsExpanded, setBuiltinsExpanded] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(false);
  const navigate = useNavigate();

  const builtinNames = useMemo(() => {
    const dummyNames = new Set(getDummyBuiltinNames());
    const iBuiltinNames = getAllIBuiltinNames().filter(
      n => !dummyNames.has(n) && !n.startsWith("__")
    );
    const allNames = new Set([...iBuiltinNames, ...SPECIAL_BUILTIN_NAMES]);
    return Array.from(allNames).sort((a, b) =>
      a.toLowerCase().localeCompare(b.toLowerCase())
    );
  }, []);

  const constantNames = useMemo(() => {
    return getAllConstantNames().sort((a, b) =>
      a.toLowerCase().localeCompare(b.toLowerCase())
    );
  }, []);

  useEffect(() => {
    loadProjects();
  }, []);

  const loadProjects = async () => {
    try {
      setLoading(true);
      const projectList = await listProjects();

      const projectsWithMetadata = await Promise.all(
        projectList.map(async project => ({
          ...project,
          fileCount: await getProjectFileCount(project.name),
          lastModified: await getProjectLastModified(project.name),
        }))
      );

      setProjects(projectsWithMetadata);
    } catch (error) {
      console.error("Failed to load projects:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteClick = (e: React.MouseEvent, project: Project) => {
    e.stopPropagation();
    setSelectedProject(project);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (!selectedProject) return;

    try {
      await deleteProject(selectedProject.name);
      await loadProjects();
      setDeleteDialogOpen(false);
      setSelectedProject(null);
    } catch (error) {
      console.error("Failed to delete project:", error);
      alert("Failed to delete project");
    }
  };

  const handleRenameClick = (e: React.MouseEvent, project: Project) => {
    e.stopPropagation();
    setSelectedProject(project);
    setNewName(project.name);
    setRenameError("");
    setRenameDialogOpen(true);
  };

  const handleRenameConfirm = async () => {
    if (!selectedProject) return;

    const existingNames = projects
      .filter(p => p.name !== selectedProject.name)
      .map(p => p.name);
    const validation = validateProjectName(newName, existingNames);

    if (!validation.valid) {
      setRenameError(validation.error || "Invalid project name");
      return;
    }

    try {
      await renameProject(selectedProject.name, newName);
      await loadProjects();
      setRenameDialogOpen(false);
      setSelectedProject(null);
      setNewName("");
      setRenameError("");
    } catch (error) {
      console.error("Failed to rename project:", error);
      setRenameError("Failed to rename project");
    }
  };

  const toggleBuiltins = useCallback(() => {
    setBuiltinsExpanded(prev => !prev);
  }, []);

  return (
    <Box
      sx={{
        maxWidth: 750,
        mx: "auto",
        px: 3,
        py: 5,
        minHeight: "100vh",
      }}
    >
      {/* Hero */}
      <Box sx={{ mb: 5, textAlign: "center" }}>
        <Typography
          variant="h3"
          component="h1"
          sx={{
            fontWeight: 800,
            letterSpacing: "-0.03em",
            mb: 0.75,
            background:
              "linear-gradient(135deg, #2563eb 0%, #7c3aed 50%, #db2777 100%)",
            backgroundClip: "text",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            fontSize: { xs: "2.5rem", sm: "3rem" },
          }}
        >
          numbl
        </Typography>
        <Typography
          variant="body1"
          color="text.secondary"
          sx={{ mb: 2.5, fontSize: "1.05rem", lineHeight: 1.5 }}
        >
          A numerical computing environment for the browser and command line
        </Typography>

        {/* Links row */}
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 1,
            mb: 1,
            flexWrap: "wrap",
          }}
        >
          {[
            {
              label: "Learn more",
              icon: <InfoOutlinedIcon sx={{ fontSize: 14 }} />,
              onClick: () => setAboutOpen(true),
            },
            {
              label: "GitHub",
              icon: <GitHubIcon sx={{ fontSize: 14 }} />,
              href: "https://github.com/flatironinstitute/numbl",
            },
            {
              label: "npm install -g numbl",
              icon: <TerminalIcon sx={{ fontSize: 14 }} />,
              href: "https://www.npmjs.com/package/numbl",
              mono: true,
            },
          ].map((link, i) => (
            <Box
              key={i}
              component={link.href ? "a" : "span"}
              {...(link.href
                ? {
                    href: link.href,
                    target: "_blank",
                    rel: "noopener noreferrer",
                  }
                : { onClick: link.onClick })}
              sx={{
                display: "inline-flex",
                alignItems: "center",
                gap: 0.5,
                cursor: "pointer",
                color: "text.secondary",
                textDecoration: "none",
                px: 1.25,
                py: 0.5,
                borderRadius: 1,
                border: "1px solid",
                borderColor: "divider",
                "&:hover": {
                  color: "primary.main",
                  borderColor: "primary.main",
                  bgcolor: "action.hover",
                },
                transition: "all 0.15s",
              }}
            >
              {link.icon}
              <Typography
                variant="caption"
                sx={
                  link.mono
                    ? { fontFamily: "monospace", fontSize: "0.7rem" }
                    : undefined
                }
              >
                {link.label}
              </Typography>
            </Box>
          ))}
        </Box>
        <Box
          onClick={toggleBuiltins}
          sx={{
            display: "inline-flex",
            alignItems: "center",
            gap: 0.5,
            cursor: "pointer",
            userSelect: "none",
            color: "text.secondary",
            "&:hover": { color: "text.primary" },
            transition: "color 0.15s",
            mt: 1,
          }}
        >
          <Typography variant="caption">
            {builtinNames.length} built-in functions and more
          </Typography>
          <ExpandMoreIcon
            sx={{
              fontSize: 16,
              transform: builtinsExpanded ? "rotate(180deg)" : "rotate(0deg)",
              transition: "transform 0.2s",
            }}
          />
        </Box>
        <Collapse in={builtinsExpanded}>
          <Box sx={{ mt: 2.5, textAlign: "left" }}>
            <CapabilitySection
              label={`Built-in functions (${builtinNames.length})`}
              items={builtinNames}
              mono
            />
            {constantNames.length > 0 && (
              <CapabilitySection
                label={`Constants (${constantNames.length})`}
                items={constantNames}
                mono
              />
            )}
            {CAPABILITIES.map(category => (
              <CapabilitySection
                key={category.label}
                label={category.label}
                items={category.items}
              />
            ))}
          </Box>
        </Collapse>
      </Box>

      {/* Projects Section */}
      <Box sx={{ mb: 4 }}>
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            mb: 2,
          }}
        >
          <Typography
            variant="subtitle2"
            color="text.secondary"
            sx={{
              textTransform: "uppercase",
              letterSpacing: "0.1em",
              fontSize: "0.7rem",
              fontWeight: 600,
            }}
          >
            Projects
          </Typography>
          <Button
            variant="contained"
            size="small"
            startIcon={<AddIcon />}
            onClick={() => setCreateDialogOpen(true)}
            sx={{
              textTransform: "none",
              fontWeight: 600,
              borderRadius: 1.5,
              boxShadow: "none",
              "&:hover": { boxShadow: "none" },
            }}
          >
            New Project
          </Button>
        </Box>

        {loading ? (
          <Box sx={{ display: "flex", justifyContent: "center", py: 6 }}>
            <CircularProgress size={28} />
          </Box>
        ) : projects.length === 0 ? (
          <Box
            sx={{
              textAlign: "center",
              py: 7,
              px: 3,
              border: "2px dashed",
              borderColor: "divider",
              borderRadius: 3,
              bgcolor: "rgba(0,0,0,0.01)",
            }}
          >
            <Typography
              variant="body1"
              sx={{ mb: 0.5, fontWeight: 600, fontSize: "1.1rem" }}
            >
              No projects yet
            </Typography>
            <Typography
              variant="body2"
              color="text.secondary"
              sx={{ mb: 3, maxWidth: 360, mx: "auto" }}
            >
              Create a project to start writing and running MATLAB-compatible
              code in your browser.
            </Typography>
            <Button
              variant="contained"
              startIcon={<AddIcon />}
              onClick={() => setCreateDialogOpen(true)}
              sx={{
                textTransform: "none",
                fontWeight: 600,
                borderRadius: 1.5,
                boxShadow: "none",
                px: 3,
                "&:hover": { boxShadow: "none" },
              }}
            >
              Create your first project
            </Button>
          </Box>
        ) : (
          <TableContainer
            sx={{
              border: "1px solid",
              borderColor: "divider",
              borderRadius: 2,
              overflow: "hidden",
            }}
          >
            <Table size="small">
              <TableHead>
                <TableRow sx={{ bgcolor: "action.hover" }}>
                  <TableCell
                    sx={{
                      fontWeight: 600,
                      fontSize: "0.75rem",
                      color: "text.secondary",
                    }}
                  >
                    Name
                  </TableCell>
                  <TableCell
                    align="right"
                    sx={{
                      fontWeight: 600,
                      fontSize: "0.75rem",
                      color: "text.secondary",
                    }}
                  >
                    Files
                  </TableCell>
                  <TableCell
                    align="right"
                    sx={{
                      fontWeight: 600,
                      fontSize: "0.75rem",
                      color: "text.secondary",
                    }}
                  >
                    Modified
                  </TableCell>
                  <TableCell align="right" sx={{ width: 80 }} />
                </TableRow>
              </TableHead>
              <TableBody>
                {projects.map(project => (
                  <TableRow
                    key={project.name}
                    hover
                    sx={{
                      cursor: "pointer",
                      "&:last-child td": { borderBottom: 0 },
                    }}
                    onClick={() => navigate(`/project/${project.name}`)}
                  >
                    <TableCell sx={{ fontWeight: 500 }}>
                      {project.displayName || project.name}
                    </TableCell>
                    <TableCell
                      align="right"
                      sx={{ color: "text.secondary", fontSize: "0.85rem" }}
                    >
                      {project.fileCount}
                    </TableCell>
                    <TableCell
                      align="right"
                      sx={{ color: "text.secondary", fontSize: "0.85rem" }}
                    >
                      {formatDate(project.lastModified || project.updatedAt)}
                    </TableCell>
                    <TableCell align="right" sx={{ whiteSpace: "nowrap" }}>
                      <Tooltip title="Rename">
                        <IconButton
                          size="small"
                          onClick={e => handleRenameClick(e, project)}
                          sx={{ opacity: 0.5, "&:hover": { opacity: 1 } }}
                        >
                          <EditIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Delete">
                        <IconButton
                          size="small"
                          onClick={e => handleDeleteClick(e, project)}
                          sx={{
                            opacity: 0.5,
                            "&:hover": { opacity: 1, color: "error.main" },
                          }}
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

      {/* Create Dialog */}
      <CreateProjectDialog
        open={createDialogOpen}
        onClose={() => setCreateDialogOpen(false)}
        onCreated={loadProjects}
      />

      {/* Delete Confirmation Dialog */}
      <Dialog
        open={deleteDialogOpen}
        onClose={() => setDeleteDialogOpen(false)}
      >
        <DialogTitle>Delete Project</DialogTitle>
        <DialogContent>
          <DialogContentText>
            Are you sure you want to delete "{selectedProject?.name}"? This will
            delete all files in the project. This action cannot be undone.
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button
            onClick={handleDeleteConfirm}
            color="error"
            variant="contained"
          >
            Delete
          </Button>
        </DialogActions>
      </Dialog>

      {/* Rename Dialog */}
      <Dialog
        open={renameDialogOpen}
        onClose={() => setRenameDialogOpen(false)}
      >
        <DialogTitle>Rename Project</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="New Name"
            type="text"
            fullWidth
            variant="outlined"
            value={newName}
            onChange={e => {
              setNewName(e.target.value);
              setRenameError("");
            }}
            error={!!renameError}
            helperText={
              renameError ||
              "Use only letters, numbers, dashes, and underscores (no spaces)"
            }
            onKeyPress={e => {
              if (e.key === "Enter") handleRenameConfirm();
            }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setRenameDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleRenameConfirm} variant="contained">
            Rename
          </Button>
        </DialogActions>
      </Dialog>

      {/* About Dialog */}
      <Dialog
        open={aboutOpen}
        onClose={() => setAboutOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle>About numbl</DialogTitle>
        <DialogContent>
          <Typography variant="body2" sx={{ mb: 2 }}>
            numbl is an open-source numerical computing environment that aims to
            be compatible with Matlab.
          </Typography>
          <Typography variant="body2" sx={{ mb: 2 }}>
            <strong>Browser</strong> — This web interface lets you create
            projects, edit files, and run code entirely in your browser. No
            server is involved; all execution happens locally. Browser execution
            is convenient but has limited functionality and is slower than the
            CLI.
          </Typography>
          <Typography variant="body2" sx={{ mb: 2 }}>
            <strong>Command line</strong> — For full performance and features,
            <a
              href="https://www.npmjs.com/package/numbl"
              target="_blank"
              rel="noopener noreferrer"
            >
              install numbl as a CLI tool
            </a>
            . It supports running scripts, an interactive REPL, and optional
            native LAPACK acceleration for linear algebra.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setAboutOpen(false)}>Close</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
