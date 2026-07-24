import { useMemo, useEffect } from "react";
import {
  Box,
  Button,
  Typography,
  List,
  ListItemButton,
  ListItemText,
} from "@mui/material";
import ArrowBackIcon from "@mui/icons-material/ArrowBack";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import OpenInNewIcon from "@mui/icons-material/OpenInNew";
import ScienceIcon from "@mui/icons-material/Science";
import { useParams, useNavigate, Link } from "react-router-dom";
import { MarkdownView } from "../components/MarkdownView";
import { docs } from "../docs/index.js";
import { usePageMeta } from "../hooks/usePageMeta";

const SIDEBAR_WIDTH = 220;

export function DocsPage() {
  const { slug } = useParams<{ slug: string }>();
  const navigate = useNavigate();

  const activeIndex = useMemo(() => {
    const idx = docs.findIndex(d => d.slug === slug);
    return idx >= 0 ? idx : 0;
  }, [slug]);
  const activeDoc = docs[activeIndex];
  const prevDoc = activeIndex > 0 ? docs[activeIndex - 1] : null;
  const nextDoc = activeIndex < docs.length - 1 ? docs[activeIndex + 1] : null;

  usePageMeta({
    title: slug ? `${activeDoc.title} — numbl` : "Documentation — numbl",
    description: activeDoc.description,
    path: slug ? `/docs/${activeDoc.slug}` : "/docs",
  });

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [activeIndex]);

  return (
    <Box sx={{ display: "flex", minHeight: "100vh" }}>
      {/* Sidebar */}
      <Box
        component="nav"
        sx={{
          width: SIDEBAR_WIDTH,
          flexShrink: 0,
          borderRight: "1px solid",
          borderColor: "divider",
          bgcolor: "background.paper",
          py: 2,
          position: "sticky",
          top: 0,
          height: "100vh",
          overflow: "auto",
          display: { xs: "none", md: "block" },
        }}
      >
        <Link to="/" style={{ textDecoration: "none" }}>
          <Typography
            variant="subtitle2"
            sx={{
              px: 2,
              pb: 1.5,
              mb: 1,
              fontWeight: 800,
              fontSize: "1.1rem",
              letterSpacing: "-0.02em",
              background:
                "linear-gradient(135deg, #2563eb 0%, #7c3aed 50%, #db2777 100%)",
              backgroundClip: "text",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              cursor: "pointer",
            }}
          >
            numbl
          </Typography>
        </Link>
        <Typography
          variant="caption"
          sx={{
            px: 2,
            pb: 1,
            display: "block",
            color: "text.secondary",
            textTransform: "uppercase",
            letterSpacing: "0.1em",
            fontSize: "0.65rem",
            fontWeight: 600,
          }}
        >
          Documentation
        </Typography>
        <List dense disablePadding>
          {docs.map(doc => (
            <ListItemButton
              key={doc.slug}
              selected={doc.slug === activeDoc.slug}
              onClick={() => navigate(`/docs/${doc.slug}`)}
              sx={{
                py: 0.5,
                px: 2,
                borderRadius: 0,
                "&.Mui-selected": {
                  bgcolor: "action.selected",
                  borderRight: "2px solid",
                  borderColor: "primary.main",
                },
              }}
            >
              <ListItemText
                primary={doc.title}
                primaryTypographyProps={{
                  fontSize: "0.82rem",
                  fontWeight: doc.slug === activeDoc.slug ? 600 : 400,
                }}
              />
            </ListItemButton>
          ))}
        </List>
        <Typography
          variant="caption"
          sx={{
            px: 2,
            pt: 2,
            pb: 1,
            display: "block",
            color: "text.secondary",
            textTransform: "uppercase",
            letterSpacing: "0.1em",
            fontSize: "0.65rem",
            fontWeight: 600,
          }}
        >
          Tools
        </Typography>
        <List dense disablePadding>
          <ListItemButton
            component="a"
            href="/test-runner/"
            target="_blank"
            rel="noopener noreferrer"
            sx={{ py: 0.5, px: 2, borderRadius: 0 }}
          >
            <ScienceIcon
              sx={{ fontSize: 16, mr: 1, color: "text.secondary" }}
            />
            <ListItemText
              primary="Browser test runner"
              primaryTypographyProps={{ fontSize: "0.82rem" }}
            />
            <OpenInNewIcon sx={{ fontSize: 12, color: "text.secondary" }} />
          </ListItemButton>
        </List>
      </Box>

      {/* Mobile nav */}
      <Box
        sx={{
          display: { xs: "flex", md: "none" },
          position: "fixed",
          top: 0,
          left: 0,
          right: 0,
          zIndex: 10,
          bgcolor: "background.paper",
          borderBottom: "1px solid",
          borderColor: "divider",
          px: 2,
          py: 1,
          gap: 1,
          overflow: "auto",
          whiteSpace: "nowrap",
        }}
      >
        {docs.map(doc => (
          <Typography
            key={doc.slug}
            component="span"
            onClick={() => navigate(`/docs/${doc.slug}`)}
            sx={{
              fontSize: "0.78rem",
              cursor: "pointer",
              fontWeight: doc.slug === activeDoc.slug ? 700 : 400,
              color:
                doc.slug === activeDoc.slug ? "primary.main" : "text.secondary",
              "&:hover": { color: "primary.main" },
              flexShrink: 0,
            }}
          >
            {doc.title}
          </Typography>
        ))}
      </Box>

      {/* Content */}
      <Box
        sx={{
          flex: 1,
          minWidth: 0,
          maxWidth: 780,
          mx: "auto",
          px: { xs: 2, sm: 4 },
          py: { xs: 8, md: 4 },
          pb: 8,
        }}
      >
        <MarkdownView source={activeDoc.content} />

        {/* Prev / Next navigation */}
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            mt: 6,
            pt: 3,
            borderTop: "1px solid",
            borderColor: "divider",
          }}
        >
          {prevDoc ? (
            <Button
              startIcon={<ArrowBackIcon />}
              onClick={() => navigate(`/docs/${prevDoc.slug}`)}
              sx={{ textTransform: "none", fontWeight: 500 }}
            >
              {prevDoc.title}
            </Button>
          ) : (
            <Box />
          )}
          {nextDoc ? (
            <Button
              endIcon={<ArrowForwardIcon />}
              onClick={() => navigate(`/docs/${nextDoc.slug}`)}
              sx={{ textTransform: "none", fontWeight: 500 }}
            >
              {nextDoc.title}
            </Button>
          ) : (
            <Box />
          )}
        </Box>
      </Box>
    </Box>
  );
}
