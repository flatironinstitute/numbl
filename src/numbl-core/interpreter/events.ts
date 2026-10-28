/**
 * Events and listeners of handle classes: `addlistener`, `listener`,
 * `notify`, property events (PreSet/PostSet/PreGet/PostGet), `events`, and
 * deleting handle objects (`delete`, `isvalid`, `ObjectBeingDestroyed`).
 *
 * Listener objects are instances of the stdlib classes `event.listener`
 * and `event.proplistener`, and event data of `event.EventData` and
 * `event.PropertyEvent` (src/stdlib/shims/+event). The registry maps each
 * source object to its listeners. A listener from `addlistener` is held by
 * the registry, so it lives as long as its source; one from `listener` is
 * not, and is removed when the last reference to it goes away (the
 * refcount reaches zero).
 */

import type { RuntimeValue } from "../runtime/types.js";
import {
  RuntimeClassInstance,
  isRuntimeCell,
  isRuntimeChar,
  isRuntimeClassInstance,
  isRuntimeClassInstanceArray,
  isRuntimeFunction,
  isRuntimeString,
} from "../runtime/types.js";
import { RTV } from "../runtime/constructors.js";
import { RuntimeError, CancellationError } from "../runtime/error.js";
import { toBool, toString } from "../runtime/convert.js";
import { incref, decref } from "../runtime/refcount.js";
import { invalidHandleError } from "../runtime/classAccess.js";
import type { EventInfo, PropertyInfo } from "../workspace/classInfo.js";
import { OBJECT_BEING_DESTROYED } from "../workspace/classTable.js";

import type { Interpreter } from "./interpreter.js";
import { instantiateClass } from "./oop.js";
import { metaPropertyFor } from "./metaClass.js";

export const EVENT_DATA_CLASS = "event.EventData";
const PROPERTY_EVENT_CLASS = "event.PropertyEvent";
const LISTENER_CLASS = "event.listener";
const PROP_LISTENER_CLASS = "event.proplistener";

export type PropertyEventName = "PreSet" | "PostSet" | "PreGet" | "PostGet";
const PROPERTY_EVENTS: readonly string[] = [
  "PreSet",
  "PostSet",
  "PreGet",
  "PostGet",
];

/** One registration of a listener object on one source object. */
interface ListenerEntry {
  listener: RuntimeClassInstance;
  /** The event name, or `Prop\0PostSet` for a property event. */
  key: string;
  /** Held by the registry (`addlistener`), so it lives with the source. */
  tied: boolean;
  /** Calls of the callback in progress, for `Recursive = false`. */
  running: number;
}

/** The listeners of a run, by source object. */
/** Event listeners (`addlistener`) live as long as the session, so a
 *  listener added by one command fires in the next. */
const sessionRegistries = new WeakMap<object, EventRegistry>();

export function sessionEventRegistry(session: object): EventRegistry {
  let reg = sessionRegistries.get(session);
  if (!reg) {
    reg = new EventRegistry();
    sessionRegistries.set(session, reg);
  }
  return reg;
}

export class EventRegistry {
  private readonly bySource = new WeakMap<
    RuntimeClassInstance,
    ListenerEntry[]
  >();
  /** The sources each listener object is registered on. */
  private readonly byListener = new WeakMap<
    RuntimeClassInstance,
    RuntimeClassInstance[]
  >();
  /** Handle objects whose deletion is in progress. */
  readonly beingDestroyed = new WeakSet<RuntimeClassInstance>();

  add(source: RuntimeClassInstance, entry: ListenerEntry): void {
    let list = this.bySource.get(source);
    if (!list) this.bySource.set(source, (list = []));
    list.push(entry);
    let sources = this.byListener.get(entry.listener);
    if (!sources) this.byListener.set(entry.listener, (sources = []));
    if (!sources.includes(source)) sources.push(source);
    if (entry.tied) incref(entry.listener);
  }

  /** The entries for `key` on `source`, newest first (MATLAB calls the
   *  most recently added listener first). A copy: listeners added while
   *  the event is being handled are not called for it. */
  entries(source: RuntimeClassInstance, key: string): ListenerEntry[] {
    const list = this.bySource.get(source);
    if (!list) return [];
    const out: ListenerEntry[] = [];
    for (let i = list.length - 1; i >= 0; i--)
      if (list[i].key === key) out.push(list[i]);
    return out;
  }

  /** Whether `entry` is still registered on `source` (not removed while
   *  an event was being handled). */
  isRegistered(source: RuntimeClassInstance, entry: ListenerEntry): boolean {
    return this.bySource.get(source)?.includes(entry) ?? false;
  }

  hasEntries(source: RuntimeClassInstance): boolean {
    return (this.bySource.get(source)?.length ?? 0) > 0;
  }

  /** Remove a listener object from every source. */
  removeListener(interp: Interpreter, listener: RuntimeClassInstance): void {
    const sources = this.byListener.get(listener);
    if (!sources) return;
    this.byListener.delete(listener);
    for (const source of sources) {
      const list = this.bySource.get(source);
      if (!list) continue;
      this.bySource.set(
        source,
        list.filter(e => e.listener !== listener)
      );
      this.release(
        interp,
        list.filter(e => e.listener === listener)
      );
    }
  }

  /** Remove every listener of a deleted source. */
  removeSource(interp: Interpreter, source: RuntimeClassInstance): void {
    const list = this.bySource.get(source);
    if (!list) return;
    this.bySource.delete(source);
    for (const e of list) {
      const sources = this.byListener.get(e.listener);
      if (sources)
        this.byListener.set(
          e.listener,
          sources.filter(s => s !== source)
        );
    }
    this.release(interp, list);
  }

  private release(interp: Interpreter, entries: ListenerEntry[]): void {
    for (const e of entries) if (e.tied) decref(interp.rt, e.listener);
  }
}

// ── Errors ──────────────────────────────────────────────────────────────

function classError(identifier: string, message: string): RuntimeError {
  return new RuntimeError(message, null, { identifier });
}

function invalidArgument(fn: string): RuntimeError {
  return classError(
    "MATLAB:class:InvalidInputArgumentTypeSizeOrValue",
    `Invalid input argument for function '${fn}'.`
  );
}

// ── Arguments ───────────────────────────────────────────────────────────

/** The handle objects of a source argument (an object or object array),
 *  each checked to be a valid handle. */
function sourceObjects(
  interp: Interpreter,
  value: RuntimeValue,
  fn: string
): RuntimeClassInstance[] {
  const elems = isRuntimeClassInstance(value)
    ? [value]
    : isRuntimeClassInstanceArray(value)
      ? value.elements
      : null;
  if (!elems || elems.some(e => !e.isHandleClass)) {
    const cls = toString(
      interp.rt.callBuiltin("class", 1, [value]) as RuntimeValue
    );
    throw classError(
      "MATLAB:UndefinedFunction",
      `Undefined function '${fn}' for input arguments of type '${cls}'.`
    );
  }
  for (const e of elems) if (!isValid(interp, e)) throw invalidHandleError();
  return elems;
}

function textArg(value: RuntimeValue, fn: string): string {
  if (isRuntimeChar(value) || isRuntimeString(value)) return toString(value);
  throw invalidArgument(fn);
}

function callbackArg(value: RuntimeValue, fn: string): RuntimeValue {
  if (isRuntimeFunction(value)) return value;
  throw invalidArgument(fn);
}

/** Property names: text, a cell array of text, or meta.property objects. */
function propertyNames(value: RuntimeValue, fn: string): string[] {
  if (isRuntimeChar(value) || isRuntimeString(value)) return [toString(value)];
  if (isRuntimeCell(value)) return value.data.map(v => textArg(v, fn));
  const metas = isRuntimeClassInstance(value)
    ? [value]
    : isRuntimeClassInstanceArray(value)
      ? value.elements
      : [];
  const names = metas.map(m => m.fields.get("Name"));
  if (metas.length > 0 && names.every(n => n !== undefined && isRuntimeChar(n)))
    return names.map(n => toString(n!));
  throw invalidArgument(fn);
}

// ── Creating listeners ──────────────────────────────────────────────────

/** A new instance of a stdlib event class, with its default values. */
function newEventObject(
  interp: Interpreter,
  className: string
): RuntimeClassInstance {
  return instantiateClass(interp, className, [], 1) as RuntimeClassInstance;
}

function setFields(
  interp: Interpreter,
  obj: RuntimeClassInstance,
  fields: [string, RuntimeValue][]
): void {
  for (const [name, value] of fields) obj.bindField(interp.rt, name, value);
}

function column(values: RuntimeValue[]): RuntimeValue {
  return RTV.cell(values, [values.length, 1]);
}

/** `addlistener(src, ...)` (tied = true) and `listener(src, ...)` /
 *  `event.listener(src, ...)` (tied = false): one listener per source
 *  object, for `(src, EventName, callback)` or `(src, PropName,
 *  PropEvent, callback)`. */
export function createListeners(
  interp: Interpreter,
  args: RuntimeValue[],
  tied: boolean,
  fn: string
): RuntimeValue {
  if (args.length !== 3 && args.length !== 4) throw invalidArgument(fn);
  const sources = sourceObjects(interp, args[0], fn);
  const table = interp.code.classes;
  const site = interp.frame.className;
  const listeners: RuntimeClassInstance[] = [];
  for (const src of sources) {
    let lh: RuntimeClassInstance;
    const keys: string[] = [];
    if (args.length === 3) {
      const name = textArg(args[1], fn);
      const callback = callbackArg(args[2], fn);
      const ev = table.findEvent(src.className, name);
      if (!ev)
        throw classError(
          "MATLAB:class:invalidEvent",
          `Event '${name}' is not defined for class '${src.className}'.`
        );
      if (!table.canAccess(ev.listenAccess, ev.definingClass, site))
        throw classError(
          "MATLAB:class:ListenRestricted",
          `Cannot listen to event '${name}' in class '${ev.definingClass}'.`
        );
      lh = newEventObject(interp, LISTENER_CLASS);
      setFields(interp, lh, [
        ["Source", RTV.cell([src], [1, 1])],
        ["EventName", RTV.char(name)],
        ["Callback", callback],
      ]);
      keys.push(name);
    } else {
      const props = propertyNames(args[1], fn);
      const eventName = textArg(args[2], fn);
      const callback = callbackArg(args[3], fn);
      if (!PROPERTY_EVENTS.includes(eventName))
        throw classError(
          "MATLAB:class:invalidEvent",
          `Event '${eventName}' is not defined for class '${src.className}'.`
        );
      const metas: RuntimeValue[] = [];
      for (const name of props) {
        const prop = observableProperty(interp, src, name, eventName);
        metas.push(metaPropertyFor(interp, prop));
        keys.push(`${name}\0${eventName}`);
      }
      lh = newEventObject(interp, PROP_LISTENER_CLASS);
      setFields(interp, lh, [
        ["Object", RTV.cell([src], [1, 1])],
        ["Source", column(metas)],
        ["EventName", RTV.char(eventName)],
        ["Callback", callback],
      ]);
    }
    for (const key of keys)
      interp.events.add(src, { listener: lh, key, tied, running: 0 });
    if (!tied) lh._onDestroy = () => interp.events.removeListener(interp, lh);
    listeners.push(lh);
  }
  if (listeners.length === 1) return listeners[0];
  return RTV.classInstanceArray(listeners[0].className, listeners, [
    listeners.length,
    1,
  ]);
}

/** The property `name` of `src`, checked to exist, be accessible and be
 *  observable for `eventName`. */
function observableProperty(
  interp: Interpreter,
  src: RuntimeClassInstance,
  name: string,
  eventName: string
): PropertyInfo {
  const table = interp.code.classes;
  const prop = table.findProperty(src.className, name);
  if (
    !prop ||
    !table.canAccess(prop.getAccess, prop.definingClass, interp.frame.className)
  )
    throw classError(
      "MATLAB:class:InvalidProperty",
      `The name '${name}' is not an accessible property for an instance of class '${src.className}'.`
    );
  const set = eventName.endsWith("Set");
  if (set ? !prop.setObservable : !prop.getObservable) {
    const kind = set ? "Set" : "Get";
    throw classError(
      `MATLAB:class:non${kind}ObservableProp`,
      `While adding a ${eventName} listener, property '${name}' in class '${src.className}' is not defined to be ${kind}Observable.`
    );
  }
  return prop;
}

// ── Triggering events ───────────────────────────────────────────────────

/** `notify(src, EventName)` and `notify(src, EventName, eventData)`. */
export function notifyEvent(interp: Interpreter, args: RuntimeValue[]): void {
  if (args.length !== 2 && args.length !== 3) throw invalidArgument("notify");
  const sources = sourceObjects(interp, args[0], "notify");
  const name = textArg(args[1], "notify");
  const table = interp.code.classes;
  let data: RuntimeClassInstance | undefined;
  if (args.length === 3) {
    const d = args[2];
    if (
      !isRuntimeClassInstance(d) ||
      !table.isSubclassOf(d.className, EVENT_DATA_CLASS)
    )
      throw invalidArgument("notify");
    data = d;
  }
  for (const src of sources) {
    const ev = table.findEvent(src.className, name);
    if (!ev)
      throw classError(
        "MATLAB:class:invalidEvent",
        `Event '${name}' is not defined for class '${src.className}'.`
      );
    if (
      !table.canAccess(
        ev.notifyAccess,
        ev.definingClass,
        interp.frame.className
      )
    )
      throw classError(
        "MATLAB:class:NotifyRestricted",
        `Cannot notify listeners of event '${name}' in class '${ev.definingClass}'.`
      );
    triggerEvent(interp, src, ev, data);
  }
}

/** Call the listeners of event `ev` on `src` with `(src, eventData)`. */
function triggerEvent(
  interp: Interpreter,
  src: RuntimeClassInstance,
  ev: EventInfo,
  data?: RuntimeClassInstance
): void {
  const entries = interp.events.entries(src, ev.name);
  if (entries.length === 0) return;
  const evt = data ?? newEventObject(interp, EVENT_DATA_CLASS);
  setFields(interp, evt, [
    ["Source", src],
    ["EventName", RTV.char(ev.name)],
  ]);
  runListeners(
    interp,
    src,
    entries,
    src,
    evt,
    `event ${ev.name} defined for class ${src.className}`
  );
}

/** A property event on `obj` (the runtime's `propertyEvent` hook). */
export function propertyEvent(
  interp: Interpreter,
  obj: RuntimeClassInstance,
  prop: PropertyInfo,
  eventName: PropertyEventName
): void {
  if (!interp.events.hasEntries(obj)) return;
  const entries = interp.events.entries(obj, `${prop.name}\0${eventName}`);
  if (entries.length === 0) return;
  const meta = metaPropertyFor(interp, prop);
  const evt = newEventObject(interp, PROPERTY_EVENT_CLASS);
  setFields(interp, evt, [
    ["AffectedObject", obj],
    ["Source", meta],
    ["EventName", RTV.char(eventName)],
  ]);
  runListeners(
    interp,
    obj,
    entries,
    meta,
    evt,
    `the ${obj.className} class ${prop.name} property ${eventName} event`
  );
}

/** Call each enabled listener's callback with `(src, evt)`. As in MATLAB,
 *  an error in a callback is reported as a warning and the remaining
 *  listeners still run; a listener whose callback is running is not
 *  called again for an event it triggers unless it is Recursive. */
function runListeners(
  interp: Interpreter,
  source: RuntimeClassInstance,
  entries: ListenerEntry[],
  src: RuntimeValue,
  evt: RuntimeClassInstance,
  /** Names the event in the warning for a failed callback. */
  what: string
): void {
  const rt = interp.rt;
  for (const entry of entries) {
    const lh = entry.listener;
    if (!isValid(interp, lh) || !interp.events.isRegistered(source, entry))
      continue;
    if (!truthy(lh.fields.get("Enabled"))) continue;
    if (entry.running > 0 && !truthy(lh.fields.get("Recursive"))) continue;
    const callback = lh.fields.get("Callback");
    if (callback === undefined) continue;
    entry.running++;
    // A compiled C callback cannot raise a catchable error.
    interp.tryDepth++;
    try {
      rt.index(callback, [src, evt], 0);
    } catch (e) {
      if (e instanceof CancellationError) throw e;
      const id = "MATLAB:callback:error";
      if (!rt.warningsOff.has(id) && !rt.warningsOff.has("all")) {
        const msg = e instanceof Error ? e.message : String(e);
        rt.output(
          `Warning: Error occurred while executing the listener callback for ${what}:\n${msg}\n`
        );
      }
    } finally {
      interp.tryDepth--;
      entry.running--;
    }
  }
}

function truthy(v: RuntimeValue | undefined): boolean {
  return v !== undefined && toBool(v);
}

// ── Deleting handles ────────────────────────────────────────────────────

/** Whether a handle object is valid (not deleted or being deleted). */
export function isValid(
  interp: Interpreter,
  obj: RuntimeClassInstance
): boolean {
  return !obj._deleted && !interp.events.beingDestroyed.has(obj);
}

/** Delete a handle object: trigger `ObjectBeingDestroyed`, run the class's
 *  delete method (`body`, when it has one), then invalidate the object and
 *  drop its listeners. Deleting a listener object removes it from its
 *  sources. A deleted object is left alone. */
export function destroyHandle(
  interp: Interpreter,
  obj: RuntimeClassInstance,
  body?: () => unknown
): unknown {
  const events = interp.events;
  if (obj._deleted || events.beingDestroyed.has(obj)) return body?.();
  events.beingDestroyed.add(obj);
  try {
    const entries = events.entries(obj, OBJECT_BEING_DESTROYED.name);
    if (entries.length > 0) {
      const evt = newEventObject(interp, EVENT_DATA_CLASS);
      setFields(interp, evt, [
        ["Source", obj],
        ["EventName", RTV.char(OBJECT_BEING_DESTROYED.name)],
      ]);
      runListeners(
        interp,
        obj,
        entries,
        obj,
        evt,
        `event ${OBJECT_BEING_DESTROYED.name} defined for class ${obj.className}`
      );
    }
    return body?.();
  } finally {
    events.beingDestroyed.delete(obj);
    obj._deleted = true;
    events.removeSource(interp, obj);
    events.removeListener(interp, obj);
  }
}

// ── events(obj) ─────────────────────────────────────────────────────────

/** The events `events(obj)` lists: those anyone may listen to that are not
 *  Hidden. */
export function listedEvents(interp: Interpreter, className: string): string[] {
  return interp.code.classes
    .allEvents(className)
    .filter(e => e.listenAccess === "public" && !e.hidden)
    .map(e => e.name);
}
