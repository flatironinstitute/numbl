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
import {
  getAllBuiltinNames,
  getAllConstantNames,
} from "../numbl-core/builtins";
import { getDummyBuiltinNames } from "../numbl-core/builtins/dummy";

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

const SPECIAL_BUILTINS = [
  "disp",
  "fprintf",
  "arrayfun",
  "cellfun",
  "structfun",
  "feval",
  "bsxfun",
  "subsref",
  "subsasgn",
  "builtin",
  "drawnow",
  "pause",
];

const CAPABILITIES: { label: string; items: string[] }[] = [
  {
    label: "Language",
    items: [
      "classes & inheritance",
      "abstract classes",
      "handle classes",
      "namespaces & packages",
      "nested functions",
      "anonymous functions",
      "closures",
      "function handles",
      "try / catch",
      "switch / case",
      "for / while loops",
      "varargin / varargout",
      "argument validation",
      "import statements",
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
      "char arrays",
      "strings",
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
      "transpose (' .')",
      "colon ranges (a:b, a:s:b)",
      "concatenation ([ ; ])",
    ],
  },
  {
    label: "Other features",
    items: [
      "2-D plotting",
      "linear algebra (SVD, QR, LU, Cholesky, eigenvalues)",
      "FFT / IFFT",
      "copy-on-write arrays",
      "column-major storage",
      "type inference",
      "compile to JavaScript",
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
    const registryNames = getAllBuiltinNames().filter(
      n => !dummyNames.has(n) && !n.startsWith("__")
    );
    const allNames = new Set([...registryNames, ...SPECIAL_BUILTINS]);
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
      }}
    >
      {/* Hero */}
      <Box sx={{ mb: 5, textAlign: "center" }}>
        <Typography
          variant="h3"
          component="h1"
          sx={{
            fontWeight: 700,
            letterSpacing: "-0.02em",
            mb: 0.5,
            background: "linear-gradient(135deg, #1976d2 0%, #9c27b0 100%)",
            backgroundClip: "text",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          numbl
        </Typography>
        <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
          A numerical computing environment for the browser and command line
        </Typography>

        {/* Links row */}
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 2,
            mb: 1,
            flexWrap: "wrap",
          }}
        >
          <Box
            onClick={() => setAboutOpen(true)}
            sx={{
              display: "inline-flex",
              alignItems: "center",
              gap: 0.5,
              cursor: "pointer",
              color: "text.secondary",
              "&:hover": { color: "primary.main" },
              transition: "color 0.15s",
            }}
          >
            <InfoOutlinedIcon sx={{ fontSize: 14 }} />
            <Typography variant="caption">Learn more</Typography>
          </Box>
          <Typography variant="caption" color="text.disabled">
            |
          </Typography>
          <Box
            component="a"
            href="https://github.com/flatironinstitute/numbl"
            target="_blank"
            rel="noopener noreferrer"
            sx={{
              display: "inline-flex",
              alignItems: "center",
              gap: 0.5,
              color: "text.secondary",
              textDecoration: "none",
              "&:hover": { color: "text.primary" },
              transition: "color 0.15s",
            }}
          >
            <GitHubIcon sx={{ fontSize: 14 }} />
            <Typography variant="caption">GitHub</Typography>
          </Box>
          <Typography variant="caption" color="text.disabled">
            |
          </Typography>
          <Box
            component="a"
            href="https://www.npmjs.com/package/numbl"
            target="_blank"
            rel="noopener noreferrer"
            sx={{
              display: "inline-flex",
              alignItems: "center",
              gap: 0.5,
              color: "text.secondary",
              textDecoration: "none",
              "&:hover": { color: "text.primary" },
              transition: "color 0.15s",
            }}
          >
            <TerminalIcon sx={{ fontSize: 14 }} />
            <Typography variant="caption">npm install -g numbl</Typography>
          </Box>
          <Typography variant="caption" color="text.disabled">
            |
          </Typography>
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
            sx={{ textTransform: "uppercase", letterSpacing: "0.08em" }}
          >
            Projects
          </Typography>
          <Button
            variant="contained"
            size="small"
            startIcon={<AddIcon />}
            onClick={() => setCreateDialogOpen(true)}
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
              py: 6,
              px: 3,
              border: "1px dashed",
              borderColor: "divider",
              borderRadius: 2,
              bgcolor: "action.hover",
            }}
          >
            <Typography variant="body1" sx={{ mb: 0.5 }}>
              Get started
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mb: 2.5 }}>
              Create a project to start writing and running code in your
              browser.
            </Typography>
            <Button
              variant="contained"
              size="small"
              startIcon={<AddIcon />}
              onClick={() => setCreateDialogOpen(true)}
            >
              Create your first project
            </Button>
          </Box>
        ) : (
          <TableContainer
            sx={{
              border: "1px solid",
              borderColor: "divider",
              borderRadius: 1,
            }}
          >
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Name</TableCell>
                  <TableCell align="right">Files</TableCell>
                  <TableCell align="right">Modified</TableCell>
                  <TableCell align="right" sx={{ width: 80 }} />
                </TableRow>
              </TableHead>
              <TableBody>
                {projects.map(project => (
                  <TableRow
                    key={project.name}
                    hover
                    sx={{ cursor: "pointer" }}
                    onClick={() => navigate(`/project/${project.name}`)}
                  >
                    <TableCell>{project.displayName || project.name}</TableCell>
                    <TableCell align="right">{project.fileCount}</TableCell>
                    <TableCell align="right">
                      {formatDate(project.lastModified || project.updatedAt)}
                    </TableCell>
                    <TableCell align="right" sx={{ whiteSpace: "nowrap" }}>
                      <Tooltip title="Rename">
                        <IconButton
                          size="small"
                          onClick={e => handleRenameClick(e, project)}
                        >
                          <EditIcon fontSize="small" />
                        </IconButton>
                      </Tooltip>
                      <Tooltip title="Delete">
                        <IconButton
                          size="small"
                          onClick={e => handleDeleteClick(e, project)}
                          sx={{ color: "error.main" }}
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
