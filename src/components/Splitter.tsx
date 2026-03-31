import { useEffect, useCallback, useRef, useState, ReactNode } from "react";
import { Box } from "@mui/material";

interface SplitterProps {
  direction: "horizontal" | "vertical";
  initialSize?: number; // initial size in pixels
  minSize?: number; // minimum size in pixels
  maxSize?: number; // maximum size in pixels
  children: [ReactNode, ReactNode]; // [first panel, second panel]
  onSizeChange?: (size: number) => void;
}

/**
 * A reusable splitter component for resizable panels.
 *
 * For vertical splitters: first child is left, second child is right
 * For horizontal splitters: first child is top, second child is bottom
 *
 * Size is always in pixels to avoid issues with percentage-based calculations.
 */
export function Splitter({
  direction,
  initialSize = 300,
  minSize = 100,
  maxSize = Infinity,
  children,
  onSizeChange,
}: SplitterProps) {
  const [size, setSize] = useState(initialSize);
  const [isDragging, setIsDragging] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);
  const dragStartPosRef = useRef<number>(0);
  const dragStartSizeRef = useRef<number>(0);

  const isVertical = direction === "vertical";

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      setIsDragging(true);
      dragStartPosRef.current = isVertical ? e.clientX : e.clientY;
      dragStartSizeRef.current = size;
    },
    [isVertical, size]
  );

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isDragging || !containerRef.current) return;

      const currentPos = isVertical ? e.clientX : e.clientY;
      const delta = currentPos - dragStartPosRef.current;
      const newSize = dragStartSizeRef.current + delta;

      // Get container size to enforce max constraint
      const containerRect = containerRef.current.getBoundingClientRect();
      const containerSize = isVertical
        ? containerRect.width
        : containerRect.height;
      const effectiveMaxSize = Math.min(maxSize, containerSize - minSize - 4); // 4px for splitter

      const clampedSize = Math.max(
        minSize,
        Math.min(effectiveMaxSize, newSize)
      );

      setSize(clampedSize);
      onSizeChange?.(clampedSize);
    },
    [isDragging, isVertical, minSize, maxSize, onSizeChange]
  );

  const handleMouseUp = useCallback(() => {
    setIsDragging(false);
  }, []);

  useEffect(() => {
    if (isDragging) {
      // Add listeners to document to handle dragging outside the component
      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", handleMouseUp);
      // Prevent text selection while dragging
      document.body.style.userSelect = "none";
      document.body.style.cursor = isVertical ? "col-resize" : "row-resize";

      return () => {
        document.removeEventListener("mousemove", handleMouseMove);
        document.removeEventListener("mouseup", handleMouseUp);
        document.body.style.userSelect = "";
        document.body.style.cursor = "";
      };
    }
  }, [isDragging, handleMouseMove, handleMouseUp, isVertical]);

  return (
    <Box
      ref={containerRef}
      sx={{
        display: "flex",
        flexDirection: isVertical ? "row" : "column",
        width: "100%",
        height: "100%",
        overflow: "hidden",
      }}
    >
      {/* First Panel */}
      <Box
        sx={{
          [isVertical ? "width" : "height"]: `${size}px`,
          [isVertical ? "minWidth" : "minHeight"]: `${size}px`,
          [isVertical ? "maxWidth" : "maxHeight"]: `${size}px`,
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {children[0]}
      </Box>

      {/* Splitter Bar */}
      <Box
        onMouseDown={handleMouseDown}
        sx={{
          [isVertical ? "width" : "height"]: "3px",
          [isVertical ? "minWidth" : "minHeight"]: "3px",
          cursor: isVertical ? "col-resize" : "row-resize",
          bgcolor: "rgba(128,128,128,0.15)",
          "&:hover": { bgcolor: "primary.main", opacity: 0.7 },
          transition: "background-color 0.15s, opacity 0.15s",
          zIndex: 1,
        }}
      />

      {/* Second Panel */}
      <Box
        sx={{
          flex: 1,
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {children[1]}
      </Box>
    </Box>
  );
}
