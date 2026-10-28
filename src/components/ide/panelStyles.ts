/** Compact tab strip used by the IDE's panels. */
export const compactTabSx = {
  minHeight: 34,
  "& .MuiTab-root": {
    textTransform: "none",
    fontWeight: 500,
    fontSize: "0.8rem",
    minHeight: 34,
    py: 0,
  },
} as const;

export const panelTabsSx = {
  borderBottom: 1,
  borderColor: "divider",
  ...compactTabSx,
} as const;
