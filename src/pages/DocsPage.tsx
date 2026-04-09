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
import { useParams, useNavigate, Link } from "react-router-dom";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import bash from "react-syntax-highlighter/dist/esm/languages/hljs/bash";
import javascript from "react-syntax-highlighter/dist/esm/languages/hljs/javascript";
import { vs2015 } from "react-syntax-highlighter/dist/esm/styles/hljs";
import { registerMatlabHighlighter } from "../utils/registerMatlabHighlighter";
import { docs } from "../docs/index.js";

registerMatlabHighlighter();
SyntaxHighlighter.registerLanguage("bash", bash);
SyntaxHighlighter.registerLanguage("javascript", javascript);
SyntaxHighlighter.registerLanguage("js", javascript);

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
        <Box
          sx={{
            "& h1": {
              fontSize: "1.8rem",
              fontWeight: 700,
              letterSpacing: "-0.02em",
              mb: 2,
              mt: 0,
            },
            "& h2": {
              fontSize: "1.25rem",
              fontWeight: 600,
              mt: 4,
              mb: 1.5,
              borderBottom: "1px solid",
              borderColor: "divider",
              pb: 0.5,
            },
            "& h3": {
              fontSize: "1rem",
              fontWeight: 600,
              mt: 3,
              mb: 1,
            },
            "& p": {
              fontSize: "0.92rem",
              lineHeight: 1.7,
              mb: 1.5,
              color: "text.primary",
            },
            "& ul, & ol": {
              pl: 3,
              mb: 1.5,
              "& li": {
                fontSize: "0.92rem",
                lineHeight: 1.7,
                mb: 0.5,
              },
            },
            "& :not(pre) > code": {
              fontFamily: 'Menlo, Monaco, "Courier New", monospace',
              fontSize: "0.84rem",
              bgcolor: "action.hover",
              px: 0.6,
              py: 0.2,
              borderRadius: 0.5,
            },
            "& table": {
              width: "100%",
              borderCollapse: "collapse",
              mb: 2,
              fontSize: "0.85rem",
            },
            "& th": {
              textAlign: "left",
              fontWeight: 600,
              borderBottom: "2px solid",
              borderColor: "divider",
              py: 1,
              px: 1.5,
            },
            "& td": {
              borderBottom: "1px solid",
              borderColor: "divider",
              py: 0.8,
              px: 1.5,
              lineHeight: 1.5,
            },
            "& a": {
              color: "primary.main",
              textDecoration: "none",
              "&:hover": { textDecoration: "underline" },
            },
            "& hr": {
              border: "none",
              borderTop: "1px solid",
              borderColor: "divider",
              my: 3,
            },
            "& blockquote": {
              borderLeft: "3px solid",
              borderColor: "primary.main",
              pl: 2,
              ml: 0,
              color: "text.secondary",
            },
          }}
        >
          <Markdown
            remarkPlugins={[remarkGfm]}
            components={{
              pre({ children }) {
                const codeEl = children as
                  | { props?: { className?: string; children?: unknown } }
                  | undefined;
                const className = codeEl?.props?.className || "";
                const match = /language-(\w+)/.exec(className);
                const lang = match?.[1] || "text";
                const code = String(codeEl?.props?.children ?? "").replace(
                  /\n$/,
                  ""
                );
                return (
                  <SyntaxHighlighter
                    language={lang}
                    style={vs2015}
                    customStyle={{
                      background: "#1e1e1e",
                      borderRadius: 8,
                      padding: "16px 20px",
                      marginBottom: 16,
                      border: "1px solid rgba(0,0,0,0.12)",
                      fontSize: "0.82rem",
                      lineHeight: 1.6,
                    }}
                    codeTagProps={{
                      style: {
                        fontFamily: 'Menlo, Monaco, "Courier New", monospace',
                      },
                    }}
                  >
                    {code}
                  </SyntaxHighlighter>
                );
              },
            }}
          >
            {activeDoc.content}
          </Markdown>
        </Box>

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
