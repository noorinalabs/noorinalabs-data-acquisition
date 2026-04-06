# Team Feedback Log

Track all feedback events here. Format:

```
## [DATE] — [FROM] → [TO] — Severity: [minor/moderate/severe]
[Feedback content]
[Action taken, if any]
```

---

## Retrospective: Phase 1 (All Waves) — 2026-04-06

### Team Performance

- **PRs merged:** 10 (8 feature PRs + 2 hotfix PRs)
- **Issues closed:** 8 (#9–#16)
- **CI health:** 1 CI failure on PR #18 (fixed in #19), all other PRs passed or had no CI (pre-workflow)
- **Total test coverage:** 480 unit tests + integration tests, all passing
- **Timeline:** All 3 waves completed in a single session (2026-04-05 to 2026-04-06)
- **Peer reviews conducted:** 0 (charter violation — reviews were assigned but not performed)

### Per-Engineer Assessments

### Tarek Mansour
- PRs: #9, #16, #18, #19
- CI failures: 1 (PR #18 exposed Neo4j auth + ValidationReport issues; fixed in same PR then edge bug in #19)
- Must-fix items received: 0
- Tech-debt items created: 0
- Assessment: Highest-throughput contributor. Delivered scaffolding (wave 1), CI/CD (wave 3), and both post-merge hotfixes. Identified and fixed a real code bug (APPEARS_IN collection ID format mismatch). The CI failure in #18 was legitimate test discovery, not carelessness. Also fixed a ruff formatting issue in the same fix branch.
- Severity: none

### Alejandra Reyes-Fuentes
- PRs: #10
- CI failures: 0 (no CI workflow existed yet)
- Must-fix items received: 0
- Tech-debt items created: 0
- Assessment: Clean delivery of models/ and utils/ shared foundations. No issues.
- Severity: none

### Ivana Horvat
- PRs: #11
- CI failures: 0 (no CI workflow existed yet)
- Must-fix items received: 0
- Tech-debt items created: 0
- Assessment: Clean delivery of resolve/ extraction. Proactively included parse dependencies needed by resolve — good judgment call that created minor overlap with Kavitha's PR #13, resolved cleanly without conflicts.
- Severity: none

### Kwesi Boateng
- PRs: #12
- CI failures: 0 (no CI workflow existed yet)
- Must-fix items received: 0
- Tech-debt items created: 0
- Assessment: Clean delivery of graph/ and enrich/ modules. No issues.
- Severity: none

### Kavitha Sundaramurthy
- PRs: #13
- CI failures: 0 (no CI workflow existed yet)
- Must-fix items received: 0
- Tech-debt items created: 0
- Assessment: Clean delivery of acquire/ and parse/ modules. Handled overlap from Ivana's parse dependency inclusion without issue.
- Severity: none

### Nikolaos Papadopoulos
- PRs: #14
- CI failures: 0 (no CI workflow existed yet)
- Must-fix items received: 0
- Tech-debt items created: 0
- Assessment: Clean delivery of pipeline orchestration and CLI extraction. No issues.
- Severity: none

### Oyunbileg Batbayar
- PRs: #15
- CI failures: 0 (no CI workflow existed yet)
- Must-fix items received: 0
- Tech-debt items created: 0
- Assessment: Delivered comprehensive test suite — 480 unit tests adapted from isnad-graph. Good coverage across all extracted modules.
- Severity: none

### Top 3 Going Well
1. **Fast, clean delivery** — All 8 planned issues completed and merged within a single session with no blocking conflicts between parallel PRs
2. **Integration tests caught real bugs** — The CI pipeline exposed both a testcontainer auth mismatch and a genuine APPEARS_IN edge loader bug (collection ID format), validating the investment in integration testing
3. **Good engineering judgment** — Ivana's proactive inclusion of parse dependencies in the resolve PR prevented a broken extraction, and the overlap with Kavitha's work resolved cleanly

### Top 3 Pain Points
1. **No peer reviews were conducted** — All PRs were merged without code review despite assignments in the plan. This is a charter violation and bypasses quality gates.
2. **CI workflow only existed for wave 3** — Waves 1 and 2 PRs had no automated checks. Bugs that could have been caught earlier were only found after the wave merge.
3. **Wave 3 sequential dependency** — Pipeline (#14) → Tests (#15) → CI/CD (#16) was correct ordering but created a bottleneck. Future phases should consider whether test extraction can begin earlier.

### Proposed Process Changes
1. **Enforce PR reviews before merge** — Rationale: Zero reviews conducted is a charter violation. Add a branch protection rule requiring at least 1 approving review, or at minimum enforce manual review assignment tracking.
2. **Add CI workflow in wave 1** — Rationale: Having CI only in wave 3 meant waves 1–2 PRs were unvalidated. Scaffolding (wave 1) should include a basic CI workflow so all subsequent PRs get checked.
3. **Pre-merge ruff/lint check** — Rationale: A formatting issue slipped through to the fix branch. Engineers should run `ruff check` and `ruff format --check` locally before pushing, or CI should enforce it from the start.
