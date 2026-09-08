---
name: audit
description: >-
  Verify that documentation still matches reality — skills, READMEs, CLAUDE.md,
  design docs, memory. Use when the user asks to audit X for accuracy or
  staleness ("audit the skills", "check the READMEs are up to date", "is
  CLAUDE.md still right"). The method is always the same: extract each factual
  claim, verify it against the code/tree, and REPORT drift — don't silently
  rewrite prose to match a guess.
---

# Audit docs against reality

Documentation drifts: a skill cites a renamed path, a README's ledger says
"landed" for code that was reverted, CLAUDE.md names a file that moved. An audit
finds that drift. The method is uniform whatever the target:

1. **Enumerate the target docs** (all skills, all READMEs, one file — whatever
   was asked).
2. **Extract each verifiable claim** — a path, a filename, a function/module
   name, a command, a count, a "this is built / this is landed" status, a
   cross-reference to another doc.
3. **Verify each claim against the tree** — `ls`/`grep`/read the actual file or
   symbol. Present tense in prose is a *claim*, not evidence.
4. **Report the drift** as a list: doc → claim → what's actually true. Propose
   the fix, but treat rewriting as a separate step the user (or a follow-up)
   approves — an audit's product is the finding, not a silent edit. Never
   "fix" a doc by guessing the new value; verify it.

Run the enumeration and the per-doc checks in parallel where you can (an
`Explore` or general-purpose agent per target class), then collate.

## Auditing `.claude/skills/`

For each `<name>/SKILL.md`:

- **Frontmatter `name:` matches the directory name** — the two diverge after a
  rename, and the skill then loads under a name nobody invokes.
- **Every path, filename, script, and command it cites still exists and still
  does what's claimed** — `git worktree`, `devtools/*.sh` targets, test
  invocations, `gh` flags. A skill that says `--delete-branch` or `-n auto`
  where the repo's rule is the opposite is drift.
- **Conventions still match CLAUDE.md and `devtools/config.sh`** — a skill
  encoding a rule (the smoke-gate selection, the job count, the
  `--delete-branch` gotcha) must agree with its authoritative source. Values
  the scripts read from `config.sh` are the ones most often quoted stale:
  check the config, not a doc that quotes it.
- **Cross-references resolve** — a skill pointing at, e.g., the `worktree`
  skill's "Sweep worktrees already merged into main" section must name a
  section that actually exists under that heading.

## Auditing READMEs

Hardest-hitting first: any README keeping a **ledger** — a table of "landed" /
"in progress" / "measured X% faster" rows — because those make precise,
checkable claims and rot silently.

- **"landed" rows name real code** — every module, class and test file cited
  exists at the path given.
- **Every "landed" row has a test pinning what it claims** — a
  measured-improvement claim with no test holding that measurement is
  unverified, whatever the row says.
- **Measured numbers are plausibly current** — spot-check a test's docstring
  baseline against the README table; they should agree.
- **"planned" rows are genuinely absent**, not quietly built (or vice versa).

## Auditing design / proposal docs

A proposal written in the present tense describes an *intent*, and reads
exactly like a description of shipped code. So check the code for each one, and
report three states rather than two: not built, partially built (does it carry
a status block saying which half exists?), and fully built — the last of which
is a doc that should be deleted rather than kept as history. Before deleting
one, grep the tree for its filename: comments and other docs cite proposals by
name, and those citations have to go with it.

## Auditing CLAUDE.md / memory

- **CLAUDE.md** — spot-check the load-bearing specifics: cited file paths
  exist, test counts and timings aren't wildly stale, and the values it quotes
  match their source (preset names and `binaryDir`s in `CMakePresets.json`,
  `PROJECT_NAME`/`SMOKE_TESTS`/`BUILD_JOBS` in `devtools/config.sh`, the
  `LIBINTX_MAX_*` defaults in the top-level `CMakeLists.txt`). It is long and
  authoritative, so flag contradictions rather than rewriting.
- The **Fork changes** section of `CLAUDE.md` is the list a rebase onto
  upstream has to reconcile; check every file it names still carries the change
  it claims.
- **Memory** — the auto-memory files live in the Claude projects dir
  (`~/.claude/projects/<slug>/memory/*.md`), **not** in the repo tree, so an
  audit that greps only the repo will wrongly conclude they're missing. Recalled
  memories reflect what was true when written; if one names a file, flag, or
  function, confirm it still exists before recommending action on it, and treat
  a memory already marked OUTDATED as a finding rather than a source.

## Output

A per-target findings list: **doc → claim → actual → suggested fix**, most
load-bearing drift first. If nothing drifted, say so plainly — "audited N
skills, all paths and conventions current" is a valid and useful result.
