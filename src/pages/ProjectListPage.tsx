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
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import { vs2015 } from "react-syntax-highlighter/dist/esm/styles/hljs";
import { registerMatlabHighlighter } from "../utils/registerMatlabHighlighter";

registerMatlabHighlighter();
import AddIcon from "@mui/icons-material/Add";
import DeleteIcon from "@mui/icons-material/Delete";
import EditIcon from "@mui/icons-material/Edit";
import GitHubIcon from "@mui/icons-material/GitHub";
import TerminalIcon from "@mui/icons-material/Terminal";
import ImageIcon from "@mui/icons-material/Image";
import MenuBookIcon from "@mui/icons-material/MenuBook";
import CodeIcon from "@mui/icons-material/Code";
import ScienceIcon from "@mui/icons-material/Science";
import { useNavigate } from "react-router-dom";
import { CreateProjectDialog } from "../components/CreateProjectDialog";
import {
  listProjects,
  deleteProject,
  renameProject,
  getProjectFileCount,
  getProjectLastModified,
  getProject,
  createProject,
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

const CHUNKIE_EXAMPLE_HASH =
  "eJztWntv4zYS_ypEgIUph_JayqtrY4G-tr0CRXG4Pdz-4aSBItM2a70qSo6dov3sNzMkJfmVR7MFDocusLFNDofD4fA3D_K3k5lKpD4ZTX47yaJUnoxO4kWdLZW8levh8FZXUTlTejFIT8RJnGeVzCqgecO-LvN7LcsRCy4GVxdMy_g6e8O-lXpZ5QW0hk3rdZaqgiV5NGW-rzLgmCQsjeZJlE3fuk876fg6q1Q8xkFvWAE98j5aSTaVM5WpSuWZxq7lSsbsPQuH_Ukw9oPBxQ2MeFhCU5aXKcduD1ra8e_Zl9QqSo_JdcED1dd1yu_0elZn_MtKpVILJOAjT5R8JEYe_Bv0rCBTpeNSVuoBJMnTSGXYnEVlqoHzBRBFaQHfhgP8Hi-yZQm_aEWyhAli_iWvPOZUyStBYwWM8oSuyjqueC-N1jQikVlPnL99WHqenf2uVsmUffPdDx8YaIrpPFlJ7JkB9wwmwg-Z8N5CJmlP9OKeeFiKSSD8h2U_UDfIRm90GlWtUPCDk5yCmGyRwCr6ciNN_2DZN5-wLI-dMkNFgpULXL3fKJk7_ZkBVovIOU-AcJ6WUnMzXsBYMbkRgfSDMxEMh26pchUldVRJBpKAtuayov1eB8AgAdMpolhy_0yciXMzaLJeI53YbPDjBshgJxfzUk35OhDrAGksI-iz1CAkbK0ZQt9vaHqVtQpSYOilykurJcsCueU1KukPldGYWsddvdJWwBq6yhWwfscAdALj7c7WNOG2_g4Q1lWOM9BMIO3pkwPM0ckrOnnRukalRGse3WmOvDjaNpDN1LwuJffg5DwgEzw9UcY1WDk3eiIy08mROUoBDKBx8b6I8wTUs6V-oeA0czPA7Lys-EL0Pkzn8hskB-vM8kz2YM5FDjadZ3hIc2eLvR9VJj-pabXoiRBoorXSTP5aRwmr1HwBC6JJ06jgpZzeJbUEohip-MTHlQr8c-Phwqs8Hp_8Lg6BWnAbqzJO5B6kfVjDiUwkC0bsGyRndcEiZojp6KGsbC7zVFblBk4Dok2U6MGLMa5FuRJGAJKNWVwhaEyCwXDsDw2i4cwAH4ReoH2kOGUwoD-Jc80rMl1vrFXmvhs7JmUGOwhkeRn7QOUQghgTCFCNyay7F4Ho3fnrrY36tVYrWTbdZe_gDh3ReXh7t0nz6XGdhyP2EezG14uokFMLswb34efdBjcih6XKt_gHHEuppGazvGTVQqJSVP26fcjmfAgrQiERKkogzHgQCIAQRo2gJmgPBgG5DjxNpjkcyWxqjtShPUR1hYfcAfYM4rpcST0wygHHQCwF8PEO7lS4v1Ph4zsVvnSnzo67fLdXZ3A-kkhrFTc-zW7Ya3bgUX969myHelBvZ_t6O3tcb2cv1dv5rZNPVUc1dz5i36kKbFkXCZoxbT-rcqZNf5GD49GA9Sqbs5afYPeqWiCur6Jyg31Vnkgw0djgUpSxvMDoiFgB64rdyepeymyb8ec4IBkFXuhW0SvpvKx42C9UHw8Mz-C44AaY-GffvmElkxD3Ff7j59WN2S6Qnvwznhj8PlCzOMm1RGyE-Ei65riIyijdJrVtA1lgO4QUZ66nKOVsixQbBhgmBpcHAjVV8VKgInUr1IC0GVV2YxppaKwunhq9J9u78RG-M_Aj8hBe_AmpHhu-czCu3MGoIAuYJtEG_DwPRSiuT_4NLR8h4AJzuz6B33Gewq_q-gQZZXJd4ZAx6_rvJR6oMRyUKpFcFxhFzfj1CcgFEr2ZCgZ6eI87BAzbsBL52TNo2ZU8EBSIh_jRu5v2js2pi5fMep0xTqqrYFGeE0IXrxYjfL4U79y04WdY_IvmPbb6lwlyBP4ubu-i8k4myVHsuxixf-V1NoVDXeTJZg5oxe0Yj61U5EwWO18DVADhdCYIf6YyzQd2Fh4OhgL_B-a_507MeXtecHJOHA4el_N9P3L-uB85f6kfubyFg3aHUc9RTV6C_zU04EkqAF4NcVAKeo3zYsPyGfqX_ViKHAX6kMMjbJgL5pPjeDAR-JXWSaV8PPkQtmeV01Lrj0A8rTRKyAA57qNy6hMooY_6DNHxG1ZKyLwxI8Qwz8poY1DvL4ien46dd0RCt5enPro3UroVDgO1zxVShp8rpgyb9LDMwXtIAUuZJTIGr4zGUUqwWm2WhUuxE13Ys1SG6PQXsoowa1Vvz8e7BBcDw5cTlXeo38zHYXh4uJ9E4E7OaIbxv99YbNcKuiIrOPE5ZASQ62MYZBlfOsYBsPvKljYg_vCDMQtYgKqkvZ5AQ1e1OA46TtlXpvhxOd5leLknKRiFxJpBV0KwdDVt9WmqN3GEPRWcynuIznL8Ck5a4xHJNKoCPo0NXaCZRGvTCAm8azSkl13SS0t62SW9tPZzYVfOaYZTYmm7Lrtdl9RFo8DZOFGMBD6jwbbPTWOmNH00zqwOeZ71kcdbJN7XHqiW-zi91-2jwUc1DmNwLVbdqSznFqwAmfZwylXpaOgVVYWAnk-MCo3ju7w5CPGo9R2Iv7IQP2aNg-xC_NVLIf7q1pWXjkL8FSYKYC5oOzbKw-h-QWumoQa1-c7KPQJlAwsO-LeyCTfcHPmV0iAu1jWBdfo3Uv8PIrU05V9cARlx48-t8XmHYPoILiNqvx6W_8bW_2dsfQ5kvmFYYUcEqqSuLEAZ1XU1d2U111XcFXLYrufjMDhfWLzFTxi02e8Pbb85GZP1Wmw2u-X-DZX7C1fqN2V-ttl0avxvWvxE0R8r-l-JJml-eNgujtvCOFcZocIhN3KJCQEGSq5ODvKKhwdY3NGi-PgFLuSL2yQqEtDOUQ_yxYj9aEjYT7JOoyxjX__nn-hFot2qHWVfJuL3k2gjS3PoilJqF1GZiJ-O2yfGQVy8xvGAaF4nUakeiIhFU8zu4OBVC7zAWUhkpCWIN_WX6M5Qw_MSVodrpCGwXkUpm7pL5OscUOP23G2dmaHJhzorftYtXXsBh2lrVG4w25nSLeQAK9J6k5qLADwHFodojmpBq6KlgpVhUe5BljmY5r1zcyXojtLrwc_hoG9S7LG7UyMye_9WgsWYVvjLR04kc-c3sjkWm4LdrmC5K-mEcXu_va04jKpW7s4QrAjsD8sGqbFACq6iqlTrA3pExl__8OGx-0Qzw9aFov1yunOzCAcVbxStLZlGJ4G949RqnkYH7g63UAjvADBWw5xz76oQS1lnzVXhAcxorwiPosYT94IkTV0fQIitG01YUk16NOEYGYxJsmlvGM_yivKYdi89u2H7-9Uzc1oAOnz5SEMFqdDIyUcC6I_cANmgdw-y6vpRyGJEe4fJ4nbxyFUEG5rtW7sjsPbuFi-xgVP1cBTY3o3YPxwRaBWOpq_zWYUursLNAeXSdVEho6yGZiyCiFbrpmphKiz-TEkQmuPdureDeOxbcL6qv-x__OtQyYi4Xadpo1CIJMMLAaEk_B0KF0SZ8Hk4HjYh1Z--XEJnGEMUBQkTXSqz9q0FohZY6D7ygUkCFUx51ofQTvgk1Bau2fcVSNZvHwG41wJPQxs9dDD23u7B1vMO4EzBTS5nRk_ChwkfligKDoxhWztnZtE8iqARz0U5xsnlIWj9wGZlnsLKbOjwS50W3lMoiFI8FwefB3vuycSw-2RiDwMZd3J6e3B4IS7-Sjj8jTj8_kxANOcVDN-cwryuMKQXqH8KJVrrJAICvD-eQjzU-g7o_WFRrx3uvp1aV7tFaV_9PBshIcNMOMDkc3FyDw3HDMKzNeu80qjNGw2YUKV84mO3wD83nSL99suJZe8oqAb4iCtfAhAcfe8wxMt3JGGzJL9nWCw2JZUi2fgwIoPkEMv2qoBIQi3dKyiLqlRTxszMoapxaJbjDqqiWXA81h-9CW3TTRNZIp-sThKyVeBWQt5HYz7Zk29DhdfhMZgZCOeAbbTlJjrYGw4IdEVwBHQNm9cgb9aKoaQeMVgaLMKkt1q40iaF053SpmZRKTtldysODnu2NORRaDVekyRaBi71btqIgLJvUgvk8-EoFAz_3ozZhA8HV4RoYZ_7AUSzHHs8jwk2GMBGMfjH_SNENzdbzN03pAz6pgZD6GE7Gt1FWssU0gW7l6eweWqGBmqV19YFrT7sGhNQGS6i3UAUALHzF0zjRt3J6AESSt-pM1mVjA91BIM0X0G6jFoVlgcgyi8kdEO-JQH9MINRDplNW4duku-GbC8TWckkj1W1Yfz7qNZaRRkryhyfcro64badC8pAsA_gQ2pY272ammICqsD57gk6b9-kI5CZvOVhH-h-xqSbGGizIYbAozoc-CXzJG7GW702r_9Y022-8E6aMzFNjnPYD4b9RsnoFkXguaz9MLZ0ogSIENB-fUzG09pm5eQQQFWdYAChEGB5Cq09kdboA-L-Tqd2nc_Oh-LH4gDg1px6dGe4baCjlGpWhwDvOttBvDY5akMO_PCbUCLst8EELOnT5w0onJvfDyjOBlcQUeCf0HDY7OdfwWMBx-bVAcdmJ-CYhMLFHOQv0f0ciUWaU0ThnTGtIkdYVVFCAQOlTISL5CD4I4EHMDuQbcH6s4wqRc3TTOClIaQwkxrHCTuvCwyxQdtNv7FKa8V2FbtmjM3H7bjpHXef2NBrkO3nMkj4ZDJJMuwvURx9XHI8ZkJOL46Xrk-KqKyTyLxAORIG7TwiHbO2pEZvbjt7CdsbCIQp0doFW20ThLsE1xhslPT2qbHyQvmFehuEolCn5ks4bChLQtl37_oHXLFhZaMD45DdMIwHzXgqE43Nj03baitGQJ5QI4ShKb4pc3m7WAnDyHxs3HtcnWyrSPSc6u9NDHkjTqIYyw_fgTf56bHXs7__Fw2sZek";

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

  const handleOpenUntitled = async () => {
    const existing = await getProject("untitled");
    if (!existing) {
      await createProject("untitled");
    }
    navigate("/project/untitled");
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
            startIcon={<CodeIcon />}
            onClick={handleOpenUntitled}
            sx={{
              textTransform: "none",
              fontWeight: 600,
              borderRadius: 2,
              px: 3,
              py: 1,
              fontSize: "0.95rem",
            }}
          >
            Open IDE
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
            {
              label: "Browser test runner",
              icon: <ScienceIcon sx={{ fontSize: 14 }} />,
              href: "/test-runner/",
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
              "No server required. IDE, interactive REPL, and embeddable components for sharing code.",
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
      <Box
        sx={{
          mb: { xs: 5, sm: 7 },
          position: "relative",
          "& pre": {
            m: 0,
            borderRadius: "8px !important",
            border: "1px solid",
            borderColor: "divider",
            padding: "20px 24px !important",
            fontSize: "0.82rem !important",
            lineHeight: "1.7 !important",
            fontFamily: 'Menlo, Monaco, "Courier New", monospace !important',
          },
          "& code": {
            fontFamily: 'Menlo, Monaco, "Courier New", monospace !important',
          },
        }}
      >
        <SyntaxHighlighter
          language="matlab"
          style={vs2015}
          customStyle={{ background: "#1e1e1e" }}
        >
          {EXAMPLE_CODE}
        </SyntaxHighlighter>
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
        <Box
          sx={{
            mt: 2,
            p: 2.5,
            borderRadius: 2,
            border: "1px solid",
            borderColor: "primary.light",
            background:
              "linear-gradient(135deg, rgba(37,99,235,0.06) 0%, rgba(124,58,237,0.06) 50%, rgba(219,39,119,0.06) 100%)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 2,
            flexWrap: "wrap",
          }}
        >
          <Box>
            <Typography
              variant="subtitle2"
              sx={{ fontWeight: 700, fontSize: "0.95rem", mb: 0.25 }}
            >
              Featured example: chunkie
            </Typography>
            <Typography
              variant="body2"
              color="text.secondary"
              sx={{ fontSize: "0.85rem", lineHeight: 1.5 }}
            >
              Boundary integral equation solver.
            </Typography>
          </Box>
          <Button
            variant="contained"
            onClick={() => navigate(`/share#${CHUNKIE_EXAMPLE_HASH}`)}
            sx={{
              textTransform: "none",
              fontWeight: 600,
              borderRadius: 1.5,
              px: 2.5,
              py: 0.75,
              fontSize: "0.9rem",
              boxShadow: "none",
              flexShrink: 0,
              "&:hover": { boxShadow: "none" },
            }}
          >
            Open chunkie example →
          </Button>
        </Box>
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
