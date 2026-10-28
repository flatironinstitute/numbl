/**
 * Events and listeners of handle classes: addlistener, listener, notify,
 * events, isvalid. The machinery is in interpreter/events.ts.
 */

import { RTV } from "../../runtime/constructors.js";
import {
  isRuntimeChar,
  isRuntimeClassInstance,
  isRuntimeClassInstanceArray,
  isRuntimeGraphicsHandle,
  isRuntimeString,
  type RuntimeValue,
} from "../../runtime/types.js";
import { toString } from "../../runtime/convert.js";
import { RuntimeError } from "../../runtime/error.js";
import { builtin } from "./builtin.js";

builtin({
  name: "addlistener",
  help: {
    signatures: [
      "el = addlistener(src, EventName, callback)",
      "el = addlistener(src, PropName, PropEvent, callback)",
    ],
    description:
      "Add a listener for an event of the handle object(s) src. callback is called as callback(src, eventData) when the event is triggered with notify. With PropName and PropEvent ('PreSet', 'PostSet', 'PreGet' or 'PostGet'), listen to a SetObservable or GetObservable property. The listener lives as long as the source object; delete(el) removes it.",
  },
  nargin: [3, 4],
  scope: true,
  apply: (args, _nargout, { scope }) =>
    scope.addListener(args, true, "addlistener"),
});

builtin({
  name: "listener",
  help: {
    signatures: [
      "el = listener(src, EventName, callback)",
      "el = listener(src, PropName, PropEvent, callback)",
    ],
    description:
      "Create a listener like addlistener, but not tied to the source object: the listener is removed when no variable holds it any more.",
  },
  nargin: [3, 4],
  scope: true,
  apply: (args, _nargout, { scope }) =>
    scope.addListener(args, false, "listener"),
});

builtin({
  name: "notify",
  help: {
    signatures: ["notify(src, EventName)", "notify(src, EventName, eventData)"],
    description:
      "Trigger an event of the handle object src: call its enabled listeners, most recently added first, with (src, eventData). eventData is an event.EventData (or subclass) object; notify sets its Source and EventName. An error in a listener callback is reported as a warning.",
  },
  nargin: [2, 3],
  nargout: 0,
  scope: true,
  apply: (args, _nargout, { scope }) => {
    scope.notify(args);
  },
});

function eventsClassName(v: RuntimeValue): string | undefined {
  if (isRuntimeClassInstance(v) || isRuntimeClassInstanceArray(v))
    return v.className;
  if (isRuntimeChar(v) || isRuntimeString(v)) return toString(v);
  return undefined;
}

builtin({
  name: "events",
  help: {
    signatures: ["events(obj)", "events(ClassName)", "names = events(...)"],
    description:
      "List the events of a class that any code may listen to (not Hidden), own events first, or return their names as a cell column.",
  },
  nargin: 1,
  mfile: true,
  scope: true,
  apply: (args, nargout, { scope, rt }) => {
    const className = eventsClassName(args[0]);
    const names = className ? scope.classEvents(className) : undefined;
    if (nargout > 0) {
      const list = names ?? [];
      return RTV.cell(
        list.map(n => RTV.char(n)),
        [list.length, 1]
      );
    }
    if (!names || names.length === 0) {
      rt.output(`No events for class ${className ?? ""}.\n`);
      return undefined;
    }
    const lines = names.map(n => `    ${n}`).join("\n");
    rt.output(`\nEvents for class ${className}:\n\n${lines}\n\n`);
    return undefined;
  },
});

builtin({
  name: "isvalid",
  help: {
    signatures: ["tf = isvalid(H)"],
    description:
      "True for each handle object of H that has not been deleted. Graphics handles are always valid.",
  },
  nargin: 1,
  apply: (args, _nargout, { rt }) => {
    const v = args[0];
    const valid = (o: Parameters<NonNullable<typeof rt.isValidHandle>>[0]) =>
      rt.isValidHandle ? rt.isValidHandle(o) : !o._deleted;
    if (isRuntimeClassInstance(v)) return RTV.logical(valid(v));
    if (isRuntimeClassInstanceArray(v)) {
      const data = new Float64Array(v.elements.map(e => (valid(e) ? 1 : 0)));
      const t = RTV.tensor(data, [v.shape[0], v.shape[1]]);
      t._isLogical = true;
      return t;
    }
    if (isRuntimeGraphicsHandle(v)) return RTV.logical(true);
    const cls = toString(rt.callBuiltin("class", 1, [v]) as RuntimeValue);
    throw new RuntimeError(
      `Undefined function 'isvalid' for input arguments of type '${cls}'.`,
      null,
      { identifier: "MATLAB:UndefinedFunction" }
    );
  },
});
