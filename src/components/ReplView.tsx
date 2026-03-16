/* eslint-disable @typescript-eslint/no-explicit-any */
import ClearIcon from "@mui/icons-material/Clear";
import { Box, Button } from "@mui/material";
import { useRef, useEffect } from "react";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import "@xterm/xterm/css/xterm.css";

const REPL_HISTORY_KEY = "numbl-repl-history";
const MAX_HISTORY_SIZE = 500;

const PS1 = "\x1b[32m>>\x1b[0m ";
const PS2 = "   ";
const PROMPT_WIDTH = 3; // Visual width of ">> " and "   "

function loadHistory(): string[] {
  try {
    const saved = localStorage.getItem(REPL_HISTORY_KEY);
    if (saved) {
      const history = JSON.parse(saved);
      if (Array.isArray(history)) {
        return history.slice(-MAX_HISTORY_SIZE);
      }
    }
  } catch (e) {
    console.warn("Failed to load REPL history:", e);
  }
  return [];
}

function saveHistory(history: string[]): void {
  try {
    const toSave = history.slice(-MAX_HISTORY_SIZE);
    localStorage.setItem(REPL_HISTORY_KEY, JSON.stringify(toSave));
  } catch (e) {
    console.warn("Failed to save REPL history:", e);
  }
}

interface TerminalMethods {
  writeOutput: (text: string, isError?: boolean) => void;
  writePrompt: () => void;
  clearTerminal: () => void;
}

interface ReplViewProps {
  onExecute: (command: string) => void;
  onClear: () => void;
  isExecuting: boolean;
  onTerminalReady?: (methods: TerminalMethods) => void;
}

export function ReplView({
  onExecute,
  onClear,
  isExecuting,
  onTerminalReady,
}: ReplViewProps) {
  const terminalRef = useRef<HTMLDivElement>(null);
  const termRef = useRef<Terminal | null>(null);
  const fitAddonRef = useRef<FitAddon | null>(null);
  const historyRef = useRef<string[]>(loadHistory());
  const isExecutingRef = useRef(isExecuting);
  const onExecuteRef = useRef(onExecute);
  const resetInputFnRef = useRef<() => void>(() => {});

  useEffect(() => {
    isExecutingRef.current = isExecuting;
  }, [isExecuting]);

  useEffect(() => {
    onExecuteRef.current = onExecute;
  }, [onExecute]);

  // Create terminal only once
  useEffect(() => {
    if (!terminalRef.current) return;
    if (termRef.current) return;

    const term = new Terminal({
      cursorBlink: true,
      fontSize: 14,
      fontFamily: 'Menlo, Monaco, "Courier New", monospace',
      theme: {
        background: "#1e1e1e",
        foreground: "#d4d4d4",
        cursor: "#d4d4d4",
      },
    });

    const fitAddon = new FitAddon();
    term.loadAddon(fitAddon);
    term.open(terminalRef.current);
    fitAddon.fit();

    termRef.current = term;
    fitAddonRef.current = fitAddon;

    // ── Multi-line input state ──────────────────────────────────────────────
    let buf: string[] = [""];
    let row = 0;
    let col = 0;
    let sRow = 0; // cursor screen-row relative to first displayed line
    const hist = historyRef.current;
    let hIdx = -1;
    let hStash = "";

    /** Redraw all buffer lines and position the cursor. */
    function display() {
      // Build a single string to avoid visible cursor flicker in xterm.js
      let out = "";
      if (sRow > 0) out += `\x1b[${sRow}A`;
      out += "\r\x1b[J"; // column 0, clear to end of screen
      for (let i = 0; i < buf.length; i++) {
        out += (i === 0 ? PS1 : PS2) + buf[i];
        if (i < buf.length - 1) out += "\r\n";
      }
      const fromEnd = buf.length - 1 - row;
      if (fromEnd > 0) out += `\x1b[${fromEnd}A`;
      out += `\x1b[${PROMPT_WIDTH + col + 1}G`; // absolute column (1-indexed)
      sRow = row;
      term.write(out);
    }

    /** Reset buffer and show a fresh prompt. */
    function resetInput() {
      buf = [""];
      row = col = sRow = 0;
      hIdx = -1;
      hStash = "";
      term.write(PS1);
    }

    // Expose resetInput to external methods
    resetInputFnRef.current = resetInput;

    /** Insert a new line at the cursor (Alt+Enter / Shift+Enter). */
    function insertNewline() {
      const after = buf[row].slice(col);
      buf[row] = buf[row].slice(0, col);
      buf.splice(row + 1, 0, after);
      row++;
      col = 0;
      display();
    }

    /** Execute the full buffer contents. */
    function exec() {
      const code = buf.join("\n").trim();
      // Move cursor below all displayed lines
      const below = buf.length - 1 - sRow;
      if (below > 0) term.write(`\x1b[${below}B`);
      term.write("\r\n");
      sRow = 0;

      if (!code) {
        resetInput();
        return;
      }

      hist.push(buf.join("\n"));
      saveHistory(hist);
      hIdx = -1;

      // Reset buffer state (prompt will be shown by writePrompt after execution)
      buf = [""];
      row = col = sRow = 0;

      onExecuteRef.current(code);
    }

    /** Insert pasted text at cursor position. */
    function handlePaste(text: string) {
      text = text.replace(/(\r\n|\r|\n)$/, ""); // strip trailing newline
      const lines = text.split(/\r\n|\r|\n/);
      if (lines.length === 1) {
        buf[row] = buf[row].slice(0, col) + lines[0] + buf[row].slice(col);
        col += lines[0].length;
      } else {
        const after = buf[row].slice(col);
        buf[row] = buf[row].slice(0, col) + lines[0];
        for (let j = 1; j < lines.length; j++) {
          buf.splice(row + j, 0, lines[j]);
        }
        row += lines.length - 1;
        col = buf[row].length;
        buf[row] += after;
      }
      display();
    }

    /** Navigate command history. */
    function navHistory(dir: number) {
      if (hist.length === 0) return;
      if (hIdx === -1 && dir > 0) return;
      if (hIdx === -1) {
        hStash = buf.join("\n");
        hIdx = hist.length - 1;
      } else {
        hIdx += dir;
        if (hIdx < 0) {
          hIdx = 0;
          return;
        }
        if (hIdx >= hist.length) {
          buf = hStash.split("\n");
          hIdx = -1;
          row = buf.length - 1;
          col = buf[row].length;
          display();
          return;
        }
      }
      buf = hist[hIdx].split("\n");
      row = buf.length - 1;
      col = buf[row].length;
      display();
    }

    /** Handle a CSI escape sequence (everything after \x1b[). */
    function handleCSI(seq: string) {
      switch (seq) {
        case "A": // Up
          if (buf.length > 1 && row > 0) {
            row--;
            col = Math.min(col, buf[row].length);
            display();
          } else {
            navHistory(-1);
          }
          break;
        case "B": // Down
          if (buf.length > 1 && row < buf.length - 1) {
            row++;
            col = Math.min(col, buf[row].length);
            display();
          } else {
            navHistory(1);
          }
          break;
        case "C": // Right
          if (col < buf[row].length) {
            col++;
            display();
          } else if (row < buf.length - 1) {
            row++;
            col = 0;
            display();
          }
          break;
        case "D": // Left
          if (col > 0) {
            col--;
            display();
          } else if (row > 0) {
            row--;
            col = buf[row].length;
            display();
          }
          break;
        case "H": // Home
        case "1~":
          col = 0;
          display();
          break;
        case "F": // End
        case "4~":
          col = buf[row].length;
          display();
          break;
        case "3~": // Delete
          if (col < buf[row].length) {
            buf[row] = buf[row].slice(0, col) + buf[row].slice(col + 1);
            display();
          } else if (row < buf.length - 1) {
            buf[row] += buf[row + 1];
            buf.splice(row + 1, 1);
            display();
          }
          break;
        case "13;2u": // Shift+Enter (kitty keyboard protocol)
          insertNewline();
          break;
      }
    }

    // Intercept Shift+Enter and Alt+Enter at DOM level
    term.attachCustomKeyEventHandler((event: KeyboardEvent) => {
      if (isExecutingRef.current) return true;
      if (
        event.type === "keydown" &&
        event.key === "Enter" &&
        (event.shiftKey || event.altKey)
      ) {
        insertNewline();
        return false; // Prevent xterm.js from processing
      }
      return true;
    });

    // Welcome message and initial prompt
    const currentTerm = term;
    setTimeout(() => {
      if (termRef.current === currentTerm) {
        currentTerm.writeln(
          "\x1b[36mWelcome to the REPL. Alt+Enter for new line, Enter to execute.\x1b[0m"
        );
        currentTerm.write("\r\n");
        resetInput();
      }
    }, 0);

    // Handle terminal input
    term.onData((data: string) => {
      if (isExecutingRef.current) return;

      // Alt+Enter: \x1b followed by \r or \n
      if (data === "\x1b\r" || data === "\x1b\n") {
        insertNewline();
        return;
      }

      // CSI escape sequences (\x1b[...)
      if (data.startsWith("\x1b[")) {
        handleCSI(data.slice(2));
        return;
      }

      // Other escape sequences (ignore)
      if (data[0] === "\x1b") return;

      // Paste: multi-character non-escape input
      if (data.length > 1) {
        handlePaste(data);
        return;
      }

      // Single character
      const code = data.charCodeAt(0);

      if (code === 13 || code === 10) {
        // Enter (CR or LF)
        exec();
      } else if (code === 127 || code === 8) {
        // Backspace
        if (col > 0) {
          buf[row] = buf[row].slice(0, col - 1) + buf[row].slice(col);
          col--;
          display();
        } else if (row > 0) {
          col = buf[row - 1].length;
          buf[row - 1] += buf[row];
          buf.splice(row, 1);
          row--;
          display();
        }
      } else if (code === 1) {
        // Ctrl+A — start of line
        col = 0;
        display();
      } else if (code === 5) {
        // Ctrl+E — end of line
        col = buf[row].length;
        display();
      } else if (code === 11) {
        // Ctrl+K — kill to end of line
        buf[row] = buf[row].slice(0, col);
        display();
      } else if (code === 21) {
        // Ctrl+U — kill to start of line
        buf[row] = buf[row].slice(col);
        col = 0;
        display();
      } else if (code === 3) {
        // Ctrl+C — cancel
        const below = buf.length - 1 - sRow;
        if (below > 0) term.write(`\x1b[${below}B`);
        term.write("\r\n^C\r\n");
        sRow = 0;
        resetInput();
      } else if (code === 12) {
        // Ctrl+L — clear screen
        term.write("\x1b[2J\x1b[H");
        sRow = 0;
        display();
      } else if (code >= 32) {
        // Printable character
        buf[row] = buf[row].slice(0, col) + data + buf[row].slice(col);
        col++;
        display();
      }
    });

    // Handle resize
    const handleResize = () => {
      fitAddon.fit();
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      term.dispose();
      termRef.current = null;
      fitAddonRef.current = null;
    };
  }, []); // Only create terminal once

  // Expose methods to parent
  useEffect(() => {
    if (termRef.current && onTerminalReady) {
      const methods: TerminalMethods = {
        writeOutput: (text: string, isError: boolean = false) => {
          if (termRef.current) {
            const formattedText = text.replace(/\n/g, "\r\n");
            if (isError) {
              termRef.current.write(`\x1b[31m${formattedText}\x1b[0m\r\n`);
            } else {
              termRef.current.write(formattedText);
            }
          }
        },
        writePrompt: () => {
          if (termRef.current) {
            termRef.current.write("\r\n");
            resetInputFnRef.current();
          }
        },
        clearTerminal: () => {
          if (termRef.current) {
            termRef.current.clear();
            termRef.current.write("\r\n");
            historyRef.current.length = 0;
            saveHistory([]);
            resetInputFnRef.current();
          }
        },
      };

      // Also attach to terminal for backward compatibility
      (termRef.current as any).writeOutput = methods.writeOutput;
      (termRef.current as any).writePrompt = methods.writePrompt;
      (termRef.current as any).clearTerminal = methods.clearTerminal;

      onTerminalReady(methods);
    }
  }, [onTerminalReady]);

  const handleClearClick = () => {
    onClear();
    if (termRef.current) {
      (termRef.current as any).clearTerminal();
    }
  };

  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        bgcolor: "#1e1e1e",
      }}
    >
      {/* Header with Clear button */}
      <Box
        sx={{
          p: 1,
          borderBottom: 1,
          borderColor: "divider",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          bgcolor: "background.paper",
        }}
      >
        <span style={{ fontWeight: 500, fontSize: "14px" }}>REPL</span>
        <Button
          size="small"
          variant="outlined"
          startIcon={<ClearIcon />}
          onClick={handleClearClick}
          disabled={isExecuting}
        >
          Clear
        </Button>
      </Box>

      {/* Terminal */}
      <Box
        ref={terminalRef}
        sx={{
          flex: 1,
          overflow: "hidden",
          "& .xterm": {
            height: "100%",
            padding: "8px",
          },
          "& .xterm-viewport::-webkit-scrollbar": {
            display: "none",
          },
          "& .xterm-viewport": {
            scrollbarWidth: "none",
          },
        }}
      />
    </Box>
  );
}

// Re-export for compatibility
export interface ReplHistoryEntry {
  id: string;
  type: "command" | "output" | "error";
  text: string;
}
