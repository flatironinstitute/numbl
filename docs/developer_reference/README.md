# Developer Reference

This folder is an internal orientation for developers and AI coding agents contributing to numbl. It is not user-facing documentation. The goal is a fast conceptual map of the system — enough shared vocabulary to make informed changes.

**Start at [overview.md](overview.md).**

## What belongs here

- Conceptual overviews of subsystems: what they are, what they do, how they connect.
- Cross-cutting concerns (type system, runtime values, error handling).
- Stable design decisions that shape how code is written.

## What does not belong here

- Exhaustive API reference — that lives alongside the code.
- Contributor how-tos and workflow (building, testing, formatting, PR process) — those live in `CONTRIBUTING.md` and `docs/agents.md`.
- End-user documentation — that lives on the docs site.
- Release notes, roadmaps, or in-progress project notes.

## Guidelines for writing reference material

1. **Be concise.** Prefer bullets to paragraphs. If a file grows past roughly two screens, split by sub-topic.
2. **Avoid file paths and directory names.** Describe components by role and responsibility, not by where they live. The source organization changes; the reference should not break when it does. Concept names that appear in code (`IBuiltin`, `LoweringContext`, `JitType`, `RuntimeTensor`, etc.) are fine — those are stable and greppable.
3. **Don't mirror the code.** Describe *what* a component does and *how* it fits into the pipeline, not every field and method. If a reader needs the exact API, they can grep.
4. **Don't duplicate.** Cross-link with relative markdown links instead of re-explaining.
5. **One topic per file.** Each file should have a narrow scope named by its topic.
6. **Keep in sync.** When a subsystem is renamed or removed in code, update the reference in the same change. A drifted reference is worse than no reference.
7. **No code dumps.** Short illustrative snippets are fine when they clarify a concept. Long extracts are not.

## Entry point and structure

`overview.md` is the entry point and should always link out to the currently existing topic files. When adding a new topic file or folder, update `overview.md` so it stays discoverable. Do not maintain a second index anywhere else — one entry point is enough.
