import type * as monaco from "monaco-editor";
import { getAllBuiltinNames } from "./numbl-core/helpers/registry.js";
import { getAllConstantNames } from "./numbl-core/helpers/constants.js";

export const numblLanguageConfig: monaco.languages.LanguageConfiguration = {
  comments: {
    lineComment: "%",
    blockComment: ["%{", "%}"],
  },
  brackets: [
    ["{", "}"],
    ["[", "]"],
    ["(", ")"],
  ],
  autoClosingPairs: [
    { open: "{", close: "}" },
    { open: "[", close: "]" },
    { open: "(", close: ")" },
    { open: '"', close: '"' },
  ],
  surroundingPairs: [
    { open: "{", close: "}" },
    { open: "[", close: "]" },
    { open: "(", close: ")" },
    { open: '"', close: '"' },
  ],
};

export function createNumblTokensProvider(): monaco.languages.IMonarchLanguage {
  // Get builtin names dynamically from numbl-core
  const builtinFunctions = getAllBuiltinNames();
  const builtinConstants = getAllConstantNames();

  return {
    defaultToken: "",

    keywords: [
      "function",
      "if",
      "else",
      "elseif",
      "for",
      "while",
      "break",
      "continue",
      "return",
      "end",
      "classdef",
      "properties",
      "methods",
      "events",
      "enumeration",
      "arguments",
      "import",
      "switch",
      "case",
      "otherwise",
      "try",
      "catch",
      "global",
      "persistent",
      "true",
      "false",
    ],

    builtinFunctions,
    builtinConstants,

    // The main tokenizer for our languages
    tokenizer: {
      root: [
        // Section markers (must be at line start)
        [/^%%.*$/, "comment.doc"],

        // Block comments
        [/%\{/, "comment", "@blockComment"],

        // Line comments
        [/%.*$/, "comment"],

        // Identifiers and keywords — push afterValue since they produce values
        [
          /[a-zA-Z_]\w*/,
          {
            cases: {
              "@keywords": { token: "keyword", next: "@afterValue" },
              "@builtinFunctions": { token: "predefined", next: "@afterValue" },
              "@builtinConstants": {
                token: "constant.language",
                next: "@afterValue",
              },
              "@default": { token: "identifier", next: "@afterValue" },
            },
          },
        ],

        // Numbers — push afterValue since they produce values
        [
          /\d+\.?\d*([eE][+-]?\d+)?/,
          { token: "number.float", next: "@afterValue" },
        ],
        [
          /\.\d+([eE][+-]?\d+)?/,
          { token: "number.float", next: "@afterValue" },
        ],

        // Strings (single and double quoted)
        [/"([^"\\]|\\.)*$/, "string.invalid"], // unterminated double-quoted string
        [/'([^'\\]|\\.)*$/, "string.invalid"], // unterminated single-quoted string
        [/"/, "string", "@doubleQuotedString"],
        [/'/, "string", "@singleQuotedString"],

        // Closing brackets produce values — push afterValue
        [/[)\]]/, { token: "@brackets", next: "@afterValue" }],
        [/\}/, { token: "@brackets", next: "@afterValue" }],

        // Opening brackets
        [/[{([]/, "@brackets"],

        // Delimiters and operators (no ' — handled in afterValue as transpose)
        [/[;,.]/, "delimiter"],
        [
          /==|~=|<=|>=|&&|\|\||\.\.\.|\.\*|\.\/|\.\\|\.\^|[=<>~+\-*/\\^&|!@?:]/,
          "operator",
        ],

        // Whitespace
        { include: "@whitespace" },
      ],

      // State after a value-producing token (identifier, number, closing
      // bracket, or string).  In this state ' is the transpose operator,
      // not a string delimiter.
      afterValue: [
        [/\.'/, "operator"], // dot-transpose (.'), stay in afterValue
        [/'/, "operator"], // transpose ('), stay in afterValue
        [/$/, { token: "", next: "@pop" }], // end of line: pop
        [/(?=[\s\S])/, { token: "", next: "@pop" }], // any other char: pop and re-process in root
      ],

      blockComment: [
        [/%\}/, "comment", "@pop"],
        [/./, "comment"],
      ],

      doubleQuotedString: [
        [/[^\\"]+/, "string"],
        [/\\./, "string.escape"],
        // Closing " produces a value (string result), switch to afterValue
        [/"/, { token: "string", switchTo: "@afterValue" }],
      ],

      singleQuotedString: [
        [/[^\\']+/, "string"],
        [/''/, "string.escape"], // escaped single quote
        // Closing ' produces a value (string result), switch to afterValue
        [/'/, { token: "string", switchTo: "@afterValue" }],
      ],

      whitespace: [[/[ \t\r\n]+/, "white"]],
    },
  };
}
