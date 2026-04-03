import { useState, useEffect, useMemo } from "react";
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
} from "@mui/material";
import AddIcon from "@mui/icons-material/Add";
import DeleteIcon from "@mui/icons-material/Delete";
import EditIcon from "@mui/icons-material/Edit";
import GitHubIcon from "@mui/icons-material/GitHub";
import TerminalIcon from "@mui/icons-material/Terminal";
import ImageIcon from "@mui/icons-material/Image";
import MenuBookIcon from "@mui/icons-material/MenuBook";
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
import { makeShareHash } from "../utils/shareUrl.js";
import type { Project } from "../db/schema";
import { getAllIBuiltinNames } from "../numbl-core/interpreter/builtins/index.js";
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

const EXAMPLE_CODE = `x = linspace(0, 4*pi, 200);
y = sin(x) .* exp(-x/10);
fprintf('Peak value: %.4f\\n', max(y));
plot(x, y, 'LineWidth', 2);
title('Damped sine wave');
xlabel('x'); ylabel('y');`;

export function ProjectListPage() {
  const [projects, setProjects] = useState<ProjectWithMetadata[]>([]);
  const [loading, setLoading] = useState(true);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [renameDialogOpen, setRenameDialogOpen] = useState(false);
  const [selectedProject, setSelectedProject] = useState<Project | null>(null);
  const [newName, setNewName] = useState("");
  const [renameError, setRenameError] = useState("");
  const navigate = useNavigate();

  const builtinCount = useMemo(() => {
    const dummyNames = new Set(getDummyBuiltinNames());
    const iBuiltinNames = getAllIBuiltinNames().filter(
      n => !dummyNames.has(n) && !n.startsWith("__")
    );
    const allNames = new Set([...iBuiltinNames, ...SPECIAL_BUILTIN_NAMES]);
    return allNames.size;
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

  return (
    <Box
      sx={{
        maxWidth: 800,
        mx: "auto",
        px: 3,
        py: { xs: 4, sm: 6 },
        minHeight: "100vh",
      }}
    >
      {/* Hero */}
      <Box sx={{ mb: { xs: 5, sm: 7 }, textAlign: "center" }}>
        <Typography
          variant="h2"
          component="h1"
          sx={{
            fontWeight: 800,
            letterSpacing: "-0.03em",
            mb: 1.5,
            background:
              "linear-gradient(135deg, #2563eb 0%, #7c3aed 50%, #db2777 100%)",
            backgroundClip: "text",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            fontSize: { xs: "3rem", sm: "3.75rem" },
          }}
        >
          numbl
        </Typography>
        <Typography
          variant="h6"
          component="p"
          color="text.secondary"
          sx={{
            mb: 1,
            fontSize: { xs: "1rem", sm: "1.15rem" },
            lineHeight: 1.6,
            fontWeight: 400,
            maxWidth: 520,
            mx: "auto",
          }}
        >
          A MATLAB-compatible numerical computing environment with{" "}
          {builtinCount} built-in functions. Runs in your browser or on the
          command line.
        </Typography>

        {/* Primary CTAs */}
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 1.5,
            mt: 3,
            mb: 1.5,
            flexWrap: "wrap",
          }}
        >
          <Button
            variant="contained"
            startIcon={<TerminalIcon />}
            onClick={() => navigate("/embed-repl")}
            sx={{
              textTransform: "none",
              fontWeight: 600,
              borderRadius: 2,
              px: 3,
              py: 1,
              fontSize: "0.95rem",
              boxShadow: "none",
              "&:hover": { boxShadow: "none" },
            }}
          >
            Open REPL
          </Button>
          <Button
            variant="outlined"
            startIcon={<MenuBookIcon />}
            onClick={() => navigate("/docs")}
            sx={{
              textTransform: "none",
              fontWeight: 600,
              borderRadius: 2,
              px: 3,
              py: 1,
              fontSize: "0.95rem",
            }}
          >
            Documentation
          </Button>
          <Button
            variant="outlined"
            startIcon={<ImageIcon />}
            onClick={() => navigate("/gallery")}
            sx={{
              textTransform: "none",
              fontWeight: 600,
              borderRadius: 2,
              px: 3,
              py: 1,
              fontSize: "0.95rem",
            }}
          >
            Plot Gallery
          </Button>
        </Box>

        {/* Secondary links */}
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            gap: 2,
            flexWrap: "wrap",
          }}
        >
          {[
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
              component="a"
              href={link.href}
              {...("internal" in link && link.internal
                ? {
                    onClick: (e: React.MouseEvent) => {
                      e.preventDefault();
                      navigate(link.href);
                    },
                  }
                : { target: "_blank", rel: "noopener noreferrer" })}
              sx={{
                display: "inline-flex",
                alignItems: "center",
                gap: 0.5,
                color: "text.secondary",
                textDecoration: "none",
                "&:hover": { color: "primary.main" },
                transition: "color 0.15s",
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
      </Box>

      {/* Feature cards */}
      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr" },
          gap: 2,
          mb: { xs: 5, sm: 7 },
        }}
      >
        {[
          {
            title: "Runs in your browser",
            description:
              "No server required. Full IDE, interactive REPL, and embeddable components for sharing code.",
          },
          {
            title: "MATLAB-compatible",
            description: `${builtinCount} built-in functions. Classes & inheritance, closures, namespaces, and more.`,
          },
          {
            title: "Rich plotting",
            description:
              "2-D and 3-D plots, bar charts, contours, surface plots, colormaps, and subplots.",
          },
          {
            title: "CLI with JIT compiler",
            description:
              "JIT compiles hot functions with type specialization. Optional native addon for LAPACK, FFTW, and C++.",
          },
        ].map(card => (
          <Box
            key={card.title}
            sx={{
              p: 2.5,
              borderRadius: 2,
              border: "1px solid",
              borderColor: "divider",
              bgcolor: "background.paper",
            }}
          >
            <Typography
              variant="subtitle2"
              sx={{ fontWeight: 700, mb: 0.5, fontSize: "0.85rem" }}
            >
              {card.title}
            </Typography>
            <Typography
              variant="body2"
              color="text.secondary"
              sx={{ lineHeight: 1.5 }}
            >
              {card.description}
            </Typography>
          </Box>
        ))}
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{
            gridColumn: "1 / -1",
            textAlign: "center",
            lineHeight: 1.6,
            fontSize: "0.82rem",
            fontStyle: "italic",
          }}
        >
          Browser mode is for development, sharing, teaching, and integrating
          scientific computing with web applications. The CLI with native addon
          is much faster. Both are currently slower than MATLAB but actively
          improving.
        </Typography>
      </Box>

      {/* Code example */}
      <Box sx={{ mb: { xs: 5, sm: 7 }, position: "relative" }}>
        <Box
          component="pre"
          sx={{
            bgcolor: "#1e1e1e",
            color: "#d4d4d4",
            borderRadius: 2,
            px: 3,
            py: 2.5,
            fontSize: "0.82rem",
            lineHeight: 1.7,
            fontFamily: 'Menlo, Monaco, "Courier New", monospace',
            overflow: "auto",
            border: "1px solid",
            borderColor: "divider",
            m: 0,
          }}
        >
          {EXAMPLE_CODE}
        </Box>
        <Button
          size="small"
          variant="contained"
          onClick={() =>
            navigate(`/share#${makeShareHash("example", EXAMPLE_CODE)}`)
          }
          sx={{
            position: "absolute",
            top: 10,
            right: 10,
            textTransform: "none",
            fontSize: "0.8rem",
            fontWeight: 600,
            borderRadius: 1.5,
            px: 2,
            py: 0.5,
            boxShadow: "none",
            "&:hover": { boxShadow: "none" },
          }}
        >
          Run
        </Button>
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
          <Box>
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
            <Typography
              variant="caption"
              color="text.secondary"
              sx={{ fontSize: "0.65rem" }}
            >
              Stored locally in your browser
            </Typography>
          </Box>
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
    </Box>
  );
}
