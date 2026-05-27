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
  // Live container size in the splitter's axis, refreshed by a
  // ResizeObserver. Used to compute the effective rendered size each
  // frame so the first panel can shrink when the window gets smaller
  // without overwriting the user's stored `size` preference.
  const [containerAxis, setContainerAxis] = useState<number | null>(null);
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

  // Track container size so the first panel can be clamped to fit when
  // the parent shrinks (e.g. window resize). Without this the
  // pixel-fixed first panel keeps its stored size and overflows the
  // splitter root's `overflow: hidden`, clipping content below the
  // visible area even though the inner scrollbar still spans the
  // original (now off-screen) height.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const update = () => {
      const rect = el.getBoundingClientRect();
      setContainerAxis(isVertical ? rect.width : rect.height);
    };
    update();
    const observer = new ResizeObserver(update);
    observer.observe(el);
    return () => observer.disconnect();
  }, [isVertical]);

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

  // Effective rendered size: clamp the stored `size` to fit the live
  // container. We DON'T `setSize(clamped)` — that would persist the
  // shrunken value to localStorage and forget the user's preference.
  // Just render at the clamped size; when the container grows back, we
  // re-expand to the original stored size.
  const effectiveSize =
    containerAxis !== null
      ? Math.max(
          minSize,
          Math.min(size, Math.max(minSize, containerAxis - minSize - 4))
        )
      : size;

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
          [isVertical ? "width" : "height"]: `${effectiveSize}px`,
          [isVertical ? "minWidth" : "minHeight"]: `${effectiveSize}px`,
          [isVertical ? "maxWidth" : "maxHeight"]: `${effectiveSize}px`,
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
