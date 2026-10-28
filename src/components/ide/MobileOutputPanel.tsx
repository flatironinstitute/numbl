import { Box, Tab, Tabs, Typography } from "@mui/material";
import type { FiguresState } from "../../numbl-core/plotting/figuresReducer.js";
import { FigureView } from "../../graphics/FigureView.js";
import type { WorkspaceRep } from "../../numblWorkerProtocol";
import { sortedHandles } from "./figureHandles";
import { AstView, GeneratedJsView } from "./OutputPanel";

const mobileTabSx = {
  minHeight: 32,
  py: 0,
  fontSize: "0.75rem",
  minWidth: 0,
  px: 1,
};

/** Compact layout's bottom pane: Output, Figures, JS and AST tabs. */
export function MobileOutputPanel({
  tab,
  onTabChange,
  output,
  figures,
  figureTab,
  onFigureTabChange,
  generatedJS,
  workspaceRep,
}: {
  tab: number;
  onTabChange: (tab: number) => void;
  output: string;
  figures: FiguresState;
  figureTab: number;
  onFigureTabChange: (tab: number) => void;
  generatedJS: string;
  workspaceRep: WorkspaceRep | null;
}) {
  const handles = sortedHandles(figures.figs);
  return (
    <Box sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <Tabs
        value={tab}
        onChange={(_, v) => onTabChange(v)}
        variant="scrollable"
        scrollButtons="auto"
        sx={{ borderBottom: 1, borderColor: "divider", minHeight: 32 }}
      >
        <Tab label="Output" sx={mobileTabSx} />
        <Tab label="Figures" sx={mobileTabSx} />
        <Tab label="JS" sx={mobileTabSx} />
        <Tab label="AST" sx={mobileTabSx} />
      </Tabs>
      <Box sx={{ flexGrow: 1, overflow: "hidden" }}>
        {tab === 0 && (
          <Box sx={{ height: "100%", overflow: "auto", p: 1 }}>
            <pre
              style={{
                margin: 0,
                fontFamily: "monospace",
                fontSize: "13px",
                whiteSpace: "pre-wrap",
              }}
            >
              {output}
            </pre>
          </Box>
        )}
        {tab === 1 && (
          <Box sx={{ height: "100%", overflow: "auto", p: 1 }}>
            {handles.length > 0 ? (
              <>
                {handles.length > 1 && (
                  <Tabs
                    value={figureTab}
                    onChange={(_, v) => onFigureTabChange(v)}
                    variant="scrollable"
                    scrollButtons="auto"
                    sx={{ minHeight: 28, mb: 0.5 }}
                  >
                    {handles.map(h => (
                      <Tab
                        key={h}
                        label={`Fig ${h}`}
                        sx={{ minHeight: 28, py: 0, fontSize: "0.7rem" }}
                      />
                    ))}
                  </Tabs>
                )}
                <FigureView figure={figures.figs[handles[figureTab]]} />
              </>
            ) : (
              <Typography variant="body2" color="text.secondary">
                No figures
              </Typography>
            )}
          </Box>
        )}
        {tab === 2 && <GeneratedJsView code={generatedJS} />}
        {tab === 3 && <AstView rep={workspaceRep} />}
      </Box>
    </Box>
  );
}
