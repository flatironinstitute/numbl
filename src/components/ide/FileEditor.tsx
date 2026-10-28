import Editor, { type OnMount } from "@monaco-editor/react";
import { useEffect, useRef } from "react";
import { fileText, isBinaryData } from "../../hooks/useProjectFiles";
import { applyNumblDarkTheme, registerNumblLanguage } from "../monacoNumbl";

type MonacoEditor = Parameters<OnMount>[0];

/**
 * Monaco editor for the active file. `content` is null until the file has
 * loaded; the editor text is replaced once per file, so typing is never
 * overwritten by the content echoed back through `onChange`.
 */
export function FileEditor({
  fileId,
  fileName,
  content,
  onChange,
}: {
  fileId: string;
  fileName: string;
  content: Uint8Array | null;
  onChange: (value: string) => void;
}) {
  const editorRef = useRef<MonacoEditor | null>(null);
  const lastFileId = useRef("");

  useEffect(() => {
    const editor = editorRef.current;
    if (!editor || !content) return;
    if (fileId === lastFileId.current) return;
    lastFileId.current = fileId;
    editor.setValue(isBinaryData(content) ? "" : fileText(content));
  }, [fileId, content]);

  const handleMount: OnMount = (editor, monaco) => {
    editorRef.current = editor;
    // Only mark as synced if the content is already loaded; otherwise the
    // effect above sets it once the load completes.
    if (content) lastFileId.current = fileId;
    registerNumblLanguage(monaco, "numbl");
    applyNumblDarkTheme(monaco);
  };

  const lower = fileName.toLowerCase();
  return (
    <Editor
      height="100%"
      language={
        fileName.endsWith(".js")
          ? "javascript"
          : lower.endsWith(".md")
            ? "markdown"
            : "numbl"
      }
      defaultValue={content ? fileText(content) : ""}
      onChange={value => onChange(value || "")}
      onMount={handleMount}
      options={{
        minimap: { enabled: false },
        fontSize: 14,
        lineNumbers: "on",
        scrollBeyondLastLine: false,
      }}
    />
  );
}
