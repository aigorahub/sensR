---
name: review-loop
description: Submit code for automated review via Command Center. Triggers build verification, quality checks, and merge readiness analysis. Iterates on feedback until merge-ready.
argument-hint: [optional description of what the code does]
---

# Review Loop Skill

You are an autonomous coding agent running a review loop. You trigger Command Center's review pipeline, wait for results, fix any blocking issues yourself, and repeat until the PR is merge-ready.

**IMPORTANT: This is a loop. You MUST keep going until the review passes or max attempts are exhausted. Do NOT stop after receiving feedback — fix the issues and re-trigger.**

## Phase 1: Setup & Context Detection

1. Resolve configuration. Check these sources in order for each value:

   **APP_URL** (Command Center endpoint):
   - Default: `https://command-center.aigora.ai`
   - Override: `CC_APP_URL` env var, or `APP_URL` in `.env.local`

   **API_SECRET** (service-to-service auth):
   - Shell env var: `CC_API_SECRET`
   - Fallback: `INTERNAL_API_SECRET` in `.env.local`
   - If neither exists, tell the user to set `CC_API_SECRET` in their shell profile and stop.

   ```bash
   # Read CC_API_SECRET from env, fall back to .env.local
   echo "${CC_API_SECRET:-$(grep INTERNAL_API_SECRET .env.local 2>/dev/null | cut -d= -f2-)}"
   ```

2. Detect repo and branch from git:
   ```bash
   git remote get-url origin 2>/dev/null | sed 's|.*github.com[:/]||;s|\.git$||'
   ```
   ```bash
   git branch --show-current
   ```

3. Find the PR number:
   ```bash
   gh pr view --json number -q .number 2>/dev/null
   ```
   If no PR exists, tell the user to open a PR first and stop.

4. Build the prompt text. Check these sources in order:
   - If the user provided `$ARGUMENTS`, use that as the prompt.
   - Otherwise, look for a plan file in `docs/plans/` or `.ai-docs/plans/` that mentions the branch name.
   - Otherwise, read the PR body: `gh pr view --json body -q .body`
   - If nothing useful is found, use a generic prompt: "Review the code changes on this branch for quality and merge readiness."

## Phase 2: Trigger the Review

Call the trigger API using `curl`. Use `python3 -c "import sys,json; ..."` for JSON parsing (jq may not be available).

```bash
curl -s -X POST "${APP_URL}/api/orchestration/trigger" \
  -H "Content-Type: application/json" \
  -H "x-internal-api-secret: ${API_SECRET}" \
  -d '{"action":"start","repo":"REPO","branch":"BRANCH","prNumber":PR_NUM,"prompt":"PROMPT_TEXT"}'
```

Extract `sessionId` from the JSON response using python3. If the response contains an error, show it and stop.

Tell the user: "Review triggered (session: SESSION_ID). Waiting for results..."

## Phase 3: Poll for Results

Poll the status endpoint. **Wait 30 seconds between polls.** Use `sleep 30` between each curl call.

```bash
curl -s "${APP_URL}/api/orchestration/status?repo=REPO&sessionId=SESSION_ID&orgId=legacy-default" \
  -H "x-internal-api-secret: ${API_SECRET}"
```

Parse the JSON response with python3. Check `state.phase`:

- `completed` → Go to Phase 4 (success)
- `failed` → Go to Phase 4 (failure)
- `executing` AND the logs mention "waiting for fix" → Go to Phase 5 (fix issues)
- Anything else → Show the current phase and latest log message, then poll again after 30s

**Keep polling until you reach one of the above terminal/actionable states.** Do NOT give up after a few polls.

## Phase 4: Terminal States

### Success (phase=completed)
Tell the user:
- The review passed
- Show the PR URL
- Show any non-blocking warnings from `state.lastMergeResult.warnings`
- The PR is ready for human review and merge

### Failure (phase=failed)
Tell the user:
- Show `state.error`
- Show what's still failing from `state.lastQualityResult` and `state.lastMergeResult`
- Suggest manual fixes if max rework attempts were exhausted

## Phase 5: Fix Blocking Issues (THE CORE LOOP)

**This is the most important phase. You are an autonomous agent — fix the issues yourself.**

1. **Extract findings** from the status response:
   - `state.lastMergeResult.blockingIssues` — array of blocking issue descriptions
   - `state.lastMergeResult.summary` — detailed markdown summary with findings
   - `state.lastQualityResult.findings` — quality check findings text

2. **Read and understand each blocking issue.** Common issue types:
   - **Code bugs** (e.g., case-sensitivity, missing error handling) → Fix the code
   - **Documentation updates** (e.g., "TODO.md needs update", "CLAUDE.md needs update") → Update the docs
   - **Skeleton regeneration** → Run `pnpm regen-skeleton` if available
   - **Test failures** → Fix the failing tests

3. **Fix each issue** using your normal code editing tools (Read, Edit, Write). Do thorough fixes — don't just add comments or TODOs.

4. **Commit and push** the fixes:
   ```bash
   git add <specific-files-you-changed>
   ```
   ```bash
   git commit -m "fix: address review feedback - <brief description>"
   ```
   ```bash
   git push
   ```

5. **Wait 3 minutes for CI and bot reviews** to process the new commit. Bot reviewers on the PR need time to analyze the push before Command Center re-reviews:
   ```bash
   sleep 180
   ```

6. **Nudge the orchestrator** to re-review:
   ```bash
   curl -s -X POST "${APP_URL}/api/orchestration/trigger" \
     -H "Content-Type: application/json" \
     -H "x-internal-api-secret: ${API_SECRET}" \
     -d '{"action":"fix_pushed","sessionId":"SESSION_ID"}'
   ```

7. **Go back to Phase 3** (poll again). The orchestrator will re-run build verification, quality check, and merge readiness on your new code.

**Repeat Phase 3→5 until the review passes or fails permanently.**

## Important Notes

- **Do NOT use `jq`** — it may not be available. Use `python3 -c "import sys,json; data=json.load(sys.stdin); print(data['key'])"` for JSON parsing.
- **Do NOT run long-lived bash loops** — poll by making individual curl calls with `sleep 30` between them.
- **Always push before nudging** — the orchestrator checks the latest commit on the branch.
- **Be thorough with fixes** — superficial fixes will just fail the next review cycle.
- **Max 3 rework cycles** — after that the orchestrator gives up. Make each fix count.
