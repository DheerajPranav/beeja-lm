---
name: beeja-lesson
description: Teach one GPT or language-model concept using the repository's current implementation, equations, tensor shapes, and a small experiment.
argument-hint: "[concept]"
arguments:
  - concept
disable-model-invocation: true
effort: high
---

Teach concept: `$concept`

Read `CLAUDE.md`, `PROJECT_STATE.md`, and the code related to this concept.

Produce a repository-grounded lesson with:

1. Intuition in plain language.
2. The minimum necessary mathematics.
3. Tensor shapes using the project's actual configuration.
4. A walkthrough of the relevant source files and functions.
5. One tiny numerical example or safe experiment.
6. Common implementation mistakes.
7. Two understanding-check questions.

When the concept is not implemented yet, explain it using the roadmap target and create no code unless the user explicitly asks for implementation.

Append a concise summary to `LEARNING_LOG.md` only when the lesson produced a new measured experiment or a meaningful project-specific insight.
