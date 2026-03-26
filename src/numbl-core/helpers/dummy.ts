/**
 * Dummy no-op built-in function names.
 * Used as placeholders for unsupported functionality.
 */

const dummyHandleFunctions = [
  "groot",
  "gcf",
  "gca",
  "shg",
  "dir",
  "newplot",
  "caxis",
  "axis",
  "odeset",
];
const returnDummyStringFunctions = [
  "pwd",
  "datestr",
  "now",
  "lastwarn",
  "mfilename",
];
const returnEmptyArrayFunctions = ["xlim", "ylim"];
const returnDummyBooleanFunctions = ["ispc", "ismac", "isunix"];
const returnDummyCellArrayFunctions = ["listfonts"];

const fileIONumFunctions = ["fopen", "fclose", "feof"];
const fileIOStringFunctions = ["fileread", "ferror"];
const fileIOUnknownFunctions = ["fgetl", "fgets"];

export function getDummyBuiltinNames(): string[] {
  return [
    ...dummyHandleFunctions,
    ...returnDummyStringFunctions,
    ...returnEmptyArrayFunctions,
    ...returnDummyBooleanFunctions,
    ...returnDummyCellArrayFunctions,
    "get",
    "set",
    "fileparts",
    "fullfile",
    "addpath",
    ...fileIONumFunctions,
    ...fileIOStringFunctions,
    ...fileIOUnknownFunctions,
    "who",
    "whos",
  ];
}
