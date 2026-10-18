# Working in this repository

## Minimal read path

1. Read PROGRESS.md. If asked to pick a task, use its single next task.
2. Read only the matching milestone in BOARDGAME_AI_PROJECT_PLAN.md.
3. Inspect relevant source/tests. Use ARCHITECTURE.md to locate needed contracts;
   read INFORMATION_BOUNDARIES.md when changing engine/agent/learning data flow.

Do not read all Markdown, archived reports, run logs or AI handoffs by default.
Use focused searches and section reads. Historical evidence is indexed in
`docs/archive/README.md`; consult it only for an affected experiment or contract.
README.md owns setup; docs/reference/COMMANDS.md owns detailed command recipes.
The ignored AI/ folder is optional local context, not the task queue.

## Work rules

- Goal: new-game plugins plus substantial GPU self-play training without rewriting
  the learning system. Preserve milestones A–E; toy experiments do not complete them.
- Use latest-policy self-play and player-visible observations. Keep baselines and
  expert/tactical labels out of training. Preserve independent game/seat memory.
- Inspect Git status; preserve unrelated work, runs, checkpoints and frozen evidence.
  Historical protocol caps/settings apply to their named runs, not all development.
- Campaigns are run by the user in their console. Prepare code/commands and use
  disposable tests; do not automatically launch production training/evaluation.
- Run checks appropriate to the change. For local Windows pytest execution, first
  read AI/LOCAL_ENVIRONMENT.md if present for the established permission workflow.
- Keep changes uncommitted unless requested. Do not push without authorization.

## Documentation discipline

PROGRESS.md: about one page, current facts and exactly one next task. Roadmap:
requirements and acceptance. Architecture: implemented contracts. README: setup
and entry commands. Results/protocols: detailed evidence, consulted on demand.
Update existing statements; do not append session logs or duplicate experiment
narratives. Distinguish implemented, tested and experimentally demonstrated behavior.

## Keep documentation small — required for future edits

Edit documentation when behavior, requirements or the next task actually changes.
Routine edits need no separate permission. Preserve the following structure:

| File | Owns | Size guide |
|---|---|---|
| AGENTS.md | AI workflow and documentation rules | At most 100 lines |
| PROGRESS.md | Current state, important blockers, exactly one next task | At most 60 lines |
| BOARDGAME_AI_PROJECT_PLAN.md | Active milestones and acceptance criteria | About 200 lines; at most 250 |
| ARCHITECTURE.md | Short overview and links to relevant contracts | At most 120 lines |
| README.md | Setup, essential entry commands and reference links | At most 150 lines |

- Replace stale statements in place. Do not append dated completion essays,
  session handoffs, command transcripts, file-hash inventories, repeated test
  counts or experiment histories to these entry documents.
- Give each fact one owning document. Link to it instead of copying it across
  progress, roadmap, architecture and README. Do not create another status file,
  roadmap, continuation prompt or summary of those same documents.
- Keep detailed technical contracts in docs/reference/ARCHITECTURE_DETAILS.md
  and command recipes in docs/reference/COMMANDS.md. Read and edit only the
  relevant sections; do not make the full references mandatory startup reading.
- Create a new report only for distinct evidence or a substantial design that
  needs a durable reference. Prefer updating an existing relevant document.
  Add a focused index link, not a narrative copy in every entry document.
- Keep historical reports out of the default reading path. Preserve their
  results, limitations and original experiment scope. Do not revive an archived
  next-task instruction or apply an old experiment's restrictions to new work.
- Before moving or renaming a document, check incoming links and code/protocol
  references. Preserve frozen protocol content and saved-run compatibility.
- Treat size limits as maintenance budgets: remove duplication or move detail
  to its owner before adding text. Do not delete requirements or evidence merely
  to meet a line count, and do not shrink the project goal to mark tasks complete.
- Before finishing a documentation edit, check affected links/anchors, whitespace,
  size budgets and that PROGRESS.md still names exactly one next task. Summarize
  validation in the chat; do not append a validation diary to project documents.

Do not weaken these rules or raise the budgets merely to accommodate your own
output. Change this documentation policy only when the user requests it.
