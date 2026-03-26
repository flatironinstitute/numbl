import { createInterface } from "readline";
import { readFileSync, writeFileSync, appendFileSync } from "fs";
import { join } from "path";
import { homedir } from "os";
import { diagnoseErrors, formatDiagnostics } from "./numbl-core/diagnostics";
import type { RuntimeValue } from "./numbl-core/runtime/index.js";
import { WorkspaceFile, NativeBridge } from "./numbl-core/workspace/types.js";
import { PlotInstruction } from "./numbl-core/executor/types.js";
import { executeCode } from "./numbl-core/executeCode.js";
import { extractMipDirectives, processMipLoad } from "./mip-directives.js";
import { scanMFiles } from "./cli.js";

const HISTORY_FILE = join(homedir(), ".numbl_history");
const HISTORY_MAX = 1000;

/** Load history entries from disk. Returns empty array on any error. */
function loadHistory(): string[] {
  try {
    const content = readFileSync(HISTORY_FILE, "utf8");
    // Each entry is stored as a single line with literal \n for multi-line commands
    return content
      .split("\n")
      .filter(line => line.length > 0)
      .map(line => line.replace(/\\n/g, "\n").replace(/\\\\/g, "\\"));
  } catch {
    return [];
  }
}

/** Append a single history entry to disk, trimming if over the limit. */
function saveHistoryEntry(entry: string, hist: string[]) {
  try {
    if (hist.length <= HISTORY_MAX) {
      // Encode newlines so each entry is one line in the file
      const encoded = entry.replace(/\\/g, "\\\\").replace(/\n/g, "\\n");
      appendFileSync(HISTORY_FILE, encoded + "\n", "utf8");
    } else {
      // Over limit — rewrite the file with the last HISTORY_MAX entries
      const trimmed = hist.slice(-HISTORY_MAX);
      const content =
        trimmed
          .map(e => e.replace(/\\/g, "\\\\").replace(/\n/g, "\\n"))
          .join("\n") + "\n";
      writeFileSync(HISTORY_FILE, content, "utf8");
    }
  } catch {
    // Silently ignore write errors
  }
}

/**
 * Interactive multi-line REPL with bracketed paste and Alt+Enter support.
 *
 * For TTY terminals: uses raw mode with a custom input handler that supports
 * multi-line editing, bracketed paste, and command history.
 *
 * For non-TTY input (pipes): falls back to simple readline-based line-by-line
 * execution.
 */
export async function runRepl(
  initialWorkspaceFiles: WorkspaceFile[],
  onDrawnow?: (instructions: PlotInstruction[]) => void,
  initialSearchPaths?: string[],
  nativeBridge?: NativeBridge,
  optimization?: number
): Promise<void> {
  let variableValues: Record<string, RuntimeValue> = {};
  let holdState = false;
  const workspaceFiles = [...initialWorkspaceFiles];
  const searchPaths = [...(initialSearchPaths ?? [])];

  /**
   * Process mip directives in the input. If directives are found, load packages
   * and add their files/paths to the persistent workspace. Returns the cleaned
   * source (directives stripped), or null if only directives (no code to execute).
   */
  function handleMipDirectives(input: string): string | null {
    const { directives, cleanedSource } = extractMipDirectives(input, "repl");
    for (const d of directives) {
      if (d.type === "load") {
        console.log(`Loading mip package: ${d.packageName}...`);
        const results = processMipLoad(d.packageName);
        for (const result of results) {
          for (const p of result.paths) {
            searchPaths.push(p);
            workspaceFiles.push(...scanMFiles(p));
          }
          console.log(
            `  Loaded ${result.packageName} (${result.paths.length} path(s) added)`
          );
        }
      }
    }
    const trimmedClean = cleanedSource.trim();
    return trimmedClean.length > 0 ? trimmedClean : null;
  }

  console.log(
    "numbl REPL — type 'exit' or press Ctrl+D to quit, Alt+Enter for new line"
  );

  // Non-TTY (piped input): simple line-by-line readline fallback
  if (!process.stdin.isTTY) {
    const rl = createInterface({
      input: process.stdin,
      output: process.stdout,
      prompt: ">> ",
    });
    rl.prompt();
    for await (const line of rl) {
      const trimmed = line.trim();
      if (trimmed === "exit" || trimmed === "quit") break;
      if (trimmed === "") {
        rl.prompt();
        continue;
      }
      try {
        const codeToRun = handleMipDirectives(trimmed);
        if (codeToRun === null) {
          rl.prompt();
          continue;
        }
        const result = executeCode(
          codeToRun,
          {
            displayResults: true,
            onOutput: (text: string) => process.stdout.write(text),
            onDrawnow,
            initialVariableValues: variableValues,
            initialHoldState: holdState,
            optimization,
          },
          workspaceFiles,
          "repl",
          searchPaths,
          nativeBridge
        );
        variableValues = result.variableValues;
        holdState = result.holdState;
        if (result.plotInstructions.length > 0 && onDrawnow) {
          onDrawnow(result.plotInstructions);
        }
      } catch (error) {
        const diags = diagnoseErrors(error, trimmed, "repl", workspaceFiles);
        console.error(formatDiagnostics(diags));
      }
      rl.prompt();
    }
    console.log("");
    process.exit(0);
  }

  // ── TTY: multi-line REPL with bracketed paste & Alt+Enter ───────────────

  const stdin = process.stdin;
  const stdout = process.stdout;

  stdin.setRawMode(true);
  stdin.resume();
  stdout.write("\x1b[?2004h"); // enable bracketed paste

  const PS1 = ">> ";
  const PS2 = "   ";

  // Input state
  let buf: string[] = [""];
  let row = 0;
  let col = 0;
  let sRow = 0; // cursor screen-row relative to first displayed line
  const hist: string[] = loadHistory();
  let hIdx = -1;
  let hStash = "";
  let hPrefix = ""; // prefix for prefix-based history search
  let pasting = false;
  let pasteBuf: string[] = [];
  let busy = false;

  /** Redraw all buffer lines and position the cursor. */
  function display() {
    if (sRow > 0) stdout.write(`\x1b[${sRow}A`);
    stdout.write("\r\x1b[J"); // column 0, clear to end of screen
    for (let i = 0; i < buf.length; i++) {
      stdout.write((i === 0 ? PS1 : PS2) + buf[i]);
      if (i < buf.length - 1) stdout.write("\r\n");
    }
    const fromEnd = buf.length - 1 - row;
    if (fromEnd > 0) stdout.write(`\x1b[${fromEnd}A`);
    const pLen = row === 0 ? PS1.length : PS2.length;
    stdout.write(`\x1b[${pLen + col + 1}G`); // absolute column (1-indexed)
    sRow = row;
  }

  /** Reset buffer and show a fresh prompt. */
  function resetInput() {
    buf = [""];
    row = col = sRow = 0;
    hIdx = -1;
    hStash = "";
    hPrefix = "";
    stdout.write(PS1);
  }

  /** Restore terminal state. */
  function cleanup() {
    stdout.write("\x1b[?2004l"); // disable bracketed paste
    stdin.setRawMode(false);
    stdin.pause();
  }

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
    if (below > 0) stdout.write(`\x1b[${below}B`);
    stdout.write("\r\n");
    sRow = 0;

    if (!code) {
      resetInput();
      return;
    }
    if (code === "exit" || code === "quit") {
      cleanup();
      console.log("");
      process.exit(0);
    }

    const entry = buf.join("\n");
    hist.push(entry);
    saveHistoryEntry(entry, hist);
    hIdx = -1;

    busy = true;
    stdin.setRawMode(false); // normal mode so \n → \r\n in output
    try {
      const codeToRun = handleMipDirectives(code);
      if (codeToRun === null) {
        stdin.setRawMode(true);
        busy = false;
        resetInput();
        return;
      }
      const result = executeCode(
        codeToRun,
        {
          displayResults: true,
          onOutput: (text: string) => stdout.write(text),
          onDrawnow,
          initialVariableValues: variableValues,
          initialHoldState: holdState,
          optimization,
        },
        workspaceFiles,
        "repl",
        searchPaths,
        nativeBridge
      );
      variableValues = result.variableValues;
      holdState = result.holdState;
      if (result.plotInstructions.length > 0 && onDrawnow) {
        onDrawnow(result.plotInstructions);
      }
    } catch (error) {
      const diags = diagnoseErrors(error, code, "repl", workspaceFiles);
      console.error(formatDiagnostics(diags));
    }
    stdin.setRawMode(true); // back to raw mode
    busy = false;
    resetInput();
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

  /** Navigate command history, with prefix-based search. */
  function navHistory(dir: number) {
    if (hist.length === 0) return;
    if (hIdx === -1 && dir > 0) return;
    if (hIdx === -1) {
      hStash = buf.join("\n");
      // Use current input as prefix for searching
      hPrefix = hStash;
      // Find the first matching entry from the end
      if (hPrefix.length > 0) {
        // Prefix search: find the last entry starting with the prefix
        for (let k = hist.length - 1; k >= 0; k--) {
          if (hist[k].startsWith(hPrefix)) {
            hIdx = k;
            break;
          }
        }
        if (hIdx === -1) return; // no match found
      } else {
        hIdx = hist.length - 1;
      }
    } else {
      // Continue searching in the given direction
      let found = false;
      if (dir < 0) {
        for (let k = hIdx - 1; k >= 0; k--) {
          if (hPrefix.length === 0 || hist[k].startsWith(hPrefix)) {
            hIdx = k;
            found = true;
            break;
          }
        }
        if (!found) return; // already at oldest match
      } else {
        for (let k = hIdx + 1; k < hist.length; k++) {
          if (hPrefix.length === 0 || hist[k].startsWith(hPrefix)) {
            hIdx = k;
            found = true;
            break;
          }
        }
        if (!found) {
          // Restore stashed input
          buf = hStash.split("\n");
          hIdx = -1;
          row = buf.length - 1;
          col = buf[row].length;
          display();
          return;
        }
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

  resetInput();

  return new Promise<void>(resolve => {
    stdin.on("data", (chunk: Buffer) => {
      if (busy) return;
      const data = chunk.toString("utf8");
      let i = 0;

      while (i < data.length) {
        if (busy) return;

        // ── Bracketed paste sequences ──
        if (data.startsWith("\x1b[200~", i)) {
          pasting = true;
          pasteBuf = [];
          i += 6;
          continue;
        }
        if (data.startsWith("\x1b[201~", i)) {
          pasting = false;
          handlePaste(pasteBuf.join(""));
          i += 6;
          continue;
        }
        if (pasting) {
          const end = data.indexOf("\x1b[201~", i);
          if (end >= 0) {
            pasteBuf.push(data.slice(i, end));
            pasting = false;
            handlePaste(pasteBuf.join(""));
            i = end + 6;
          } else {
            pasteBuf.push(data.slice(i));
            i = data.length;
          }
          continue;
        }

        // ── Alt+Enter: \x1b followed by \r or \n ──
        if (
          data[i] === "\x1b" &&
          i + 1 < data.length &&
          (data[i + 1] === "\r" || data[i + 1] === "\n")
        ) {
          insertNewline();
          i += 2;
          continue;
        }

        // ── CSI escape sequences (\x1b[...) ──
        if (data.startsWith("\x1b[", i)) {
          let j = i + 2;
          while (
            j < data.length &&
            (data.charCodeAt(j) < 0x40 || data.charCodeAt(j) > 0x7e)
          )
            j++;
          if (j >= data.length) {
            i = data.length;
            continue;
          }
          handleCSI(data.slice(i + 2, j + 1));
          i = j + 1;
          continue;
        }

        // ── Other escape sequences (skip) ──
        if (data[i] === "\x1b") {
          i++;
          continue;
        }

        // ── Control & printable characters ──
        const code = data.charCodeAt(i);
        i++;

        if (code === 13 || code === 10) {
          // Enter (CR or LF) — execute
          try {
            exec();
          } catch (err) {
            console.error("Unexpected REPL error:", err);
            busy = false;
            resetInput();
          }
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
          if (below > 0) stdout.write(`\x1b[${below}B`);
          stdout.write("\r\n^C\r\n");
          sRow = 0;
          resetInput();
        } else if (code === 4) {
          // Ctrl+D — exit when empty
          if (buf.length === 1 && buf[0] === "") {
            stdout.write("\r\n");
            cleanup();
            resolve();
            return;
          }
        } else if (code === 12) {
          // Ctrl+L — clear screen
          stdout.write("\x1b[2J\x1b[H");
          sRow = 0;
          display();
        } else if (code >= 32) {
          // Printable character
          buf[row] = buf[row].slice(0, col) + data[i - 1] + buf[row].slice(col);
          col++;
          display();
        }
      }
    });
  });
}
