# Reusable implementation prompt

Copy the text below into a new chat, or ask the AI to read and follow this file.

---

Read AGENTS.md and PROGRESS.md, then only the relevant sections of
BOARDGAME_AI_PROJECT_PLAN.md. Follow the repository's minimal reading path and
documentation rules. Do not load all reference documents or historical reports.

Select the next meaningful task, or a small set of closely related tasks, that
advances the active roadmap. Use the next task in PROGRESS.md as the starting point,
but verify it against the current code, dependencies and roadmap acceptance criteria.
Do not repeat completed work or choose optional historical research over the active
delivery milestones.

Choose a coherent deliverable that can be implemented and verified end to end.
Do not stop at an arbitrary partial change, a plan, a scaffold or a list of future
steps when the implementation is feasible. If the next task is an audit or design,
complete it and continue into the smallest well-supported implementation that follows,
unless a material unresolved decision genuinely requires my input. Do not attempt
unrelated milestones merely to increase the number of completed checkboxes.

Briefly state what you selected, why it is next, and what will count as done. Then
do the work. Make routine decisions autonomously; ask only for missing information
that materially blocks correct implementation. Preserve unrelated changes and
existing experiments. Do not start production training campaigns or commit/push
unless I explicitly authorize that in this chat.

Keep the goal in view: a reusable platform where a new game can be added through
documented contracts and a substantial model can learn through self-play on my GPU,
without rewriting the training system. Tiny CPU fixtures and CUDA optimizer support
alone do not satisfy the large-model, batched GPU training requirements.

Inspect relevant source and tests, implement the complete selected deliverable, and
run checks appropriate to the change. Distinguish implementation correctness from
experimental learning or performance evidence. Do not claim unmeasured strength,
throughput or milestone completion. If blocked, finish independent work and state
the concrete blocker and remaining work honestly.

Update existing roadmap checkboxes only where their acceptance criteria are met.
Keep PROGRESS.md concise and replace its next-task section with exactly one next
task. Update technical contracts or commands only where behavior changed. Do not
append session histories or create another status report. Leave this reusable
prompt unchanged unless I ask to change it.

Finish with a short account of what was delivered, how it was verified, any remaining
limitations, and the next task. Include exact commands when I need to run something.
