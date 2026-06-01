import { Box } from "@mui/material";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Light as SyntaxHighlighter } from "react-syntax-highlighter";
import bash from "react-syntax-highlighter/dist/esm/languages/hljs/bash";
import javascript from "react-syntax-highlighter/dist/esm/languages/hljs/javascript";
import { vs2015 } from "react-syntax-highlighter/dist/esm/styles/hljs";
import { registerMatlabHighlighter } from "../utils/registerMatlabHighlighter";

registerMatlabHighlighter();
SyntaxHighlighter.registerLanguage("bash", bash);
SyntaxHighlighter.registerLanguage("javascript", javascript);
SyntaxHighlighter.registerLanguage("js", javascript);

// Code-fence language aliases → registered highlight.js language.
const LANG_ALIAS: Record<string, string> = {
  m: "matlab",
  numbl: "matlab",
  octave: "matlab",
};

/** Styling for rendered Markdown — shared by the docs page and the IDE. */
const MARKDOWN_SX = {
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
  "& h3": { fontSize: "1rem", fontWeight: 600, mt: 3, mb: 1 },
  "& p": {
    fontSize: "0.92rem",
    lineHeight: 1.7,
    mb: 1.5,
    color: "text.primary",
  },
  "& ul, & ol": {
    pl: 3,
    mb: 1.5,
    "& li": { fontSize: "0.92rem", lineHeight: 1.7, mb: 0.5 },
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
  "& img": { maxWidth: "100%" },
} as const;

export interface MarkdownViewProps {
  source: string;
}

/**
 * Renders Markdown (GFM) with syntax-highlighted code fences. Used by the docs
 * page and the IDE's rendered-Markdown view. The caller controls sizing/scroll.
 */
export function MarkdownView({ source }: MarkdownViewProps) {
  return (
    <Box sx={MARKDOWN_SX}>
      <Markdown
        remarkPlugins={[remarkGfm]}
        components={{
          pre({ children }) {
            const codeEl = children as
              | { props?: { className?: string; children?: unknown } }
              | undefined;
            const className = codeEl?.props?.className || "";
            const match = /language-(\w+)/.exec(className);
            const raw = match?.[1] || "text";
            const lang = LANG_ALIAS[raw] || raw;
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
        {source}
      </Markdown>
    </Box>
  );
}
