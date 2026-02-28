---
name: develop
description: Autonomous development agent. Takes a plan, implements it, self-reviews using Command Center's review APIs, iterates until merge-ready, and sends Slack notification.
argument-hint: <plan text or path to plan file>
---

# Autonomous Development Skill

You are an autonomous development agent. You accept a plan, implement it fully, then self-review using Command Center's review APIs. You iterate on feedback until all gates pass, then notify via Slack.

**IMPORTANT: You MUST keep going until all review gates pass. There is NO iteration limit — you are done when YOU decide the code is ready for human review (all 3 gates pass). Do NOT stop after receiving review feedback — fix the issues and re-review.**

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
   echo "${CC_API_SECRET:-$(grep INTERNAL_API_SECRET .env.local 2>/dev/null | cut -d= -f2-)}"
   ```

2. Detect repo from git:
   ```bash
   git remote get-url origin 2>/dev/null | sed 's|.*github.com[:/]||;s|\.git$||'
   ```

3. Read the plan from `$ARGUMENTS`:
   - If `$ARGUMENTS` looks like a file path (ends in `.md`, contains `/`, or starts with `.`), read the file contents and use that as the plan.
   - Otherwise, use `$ARGUMENTS` directly as the plan text.
   - If `$ARGUMENTS` is empty, tell the user to provide a plan and stop.

4. Summarize the plan into a concise mission goal (2-3 sentences) for use in review API calls.

## Phase 2: Branch & PR Setup

1. Create a feature branch from the current branch:
   ```bash
   git checkout -b feat/<descriptive-name-from-plan>
   ```

2. Make an initial commit (can be empty or a small scaffold) and push:
   ```bash
   git push -u origin HEAD
   ```

3. Open a **draft PR** early so PR-dependent review checks can work:
   ```bash
   gh pr create --draft --title "<concise title from plan>" --body "<plan summary>"
   ```

4. Capture the PR number:
   ```bash
   gh pr view --json number -q .number
   ```

## Phase 3: Development

Implement the plan using your normal coding tools (Read, Edit, Write, Bash).

- Work through each item in the plan systematically.
- Commit and push after each meaningful chunk of work.
- Use descriptive commit messages that reference what plan item is being addressed.
- Run any available tests locally before pushing.

After all plan items are implemented, push all remaining changes:
```bash
git add <specific-files>
git commit -m "feat: <description of changes>"
git push
```

## Phase 4: Review Loop

Before calling any review APIs, do a **self-assessment**: review the plan and confirm all items are implemented. If something is missing, go back to Phase 3.

Then run the three review gates in order. All three must pass to exit the loop. **There is no iteration limit** — keep fixing and re-reviewing until all gates pass.

### Gate 1: Code Quality

```bash
curl -s -X POST "${APP_URL}/api/review/code-quality" \
  -H "Content-Type: application/json" \
  -H "x-internal-api-secret: ${API_SECRET}" \
  -d '{"repo":"REPO","branch":"BRANCH","missionGoal":"MISSION_GOAL"}'
```

Parse with python3:
```bash
python3 -c "
import sys, json
data = json.load(sys.stdin)
if not data.get('success'):
    print('ERROR:', data.get('error', 'Unknown error'))
    sys.exit(1)
r = data['result']
print('PASSED:', r.get('passed', False))
print('ASSESSMENT:', r.get('overallAssessment', 'unknown'))
print('---FINDINGS---')
findings = r.get('findings', '')
if isinstance(findings, str):
    for f in findings.split('\\n\\n'):
        if f.strip(): print(f); print()
print('---AGENT_INSTRUCTIONS---')
print(r.get('agentInstructions', 'None'))
"
```

**Pass criteria:** `passed: true`
**On failure:** Fix issues using the `findings` and `agentInstructions` from the response. Commit, push, and re-run this gate.

### Gate 2: PR Review

```bash
curl -s -X POST "${APP_URL}/api/review/pr-review" \
  -H "Content-Type: application/json" \
  -H "x-internal-api-secret: ${API_SECRET}" \
  -d '{"repo":"REPO","prNumber":PR_NUM,"missionGoal":"MISSION_GOAL"}'
```

Parse with python3:
```bash
python3 -c "
import sys, json
data = json.load(sys.stdin)
if not data.get('success'):
    print('ERROR:', data.get('error', 'Unknown error'))
    sys.exit(1)
r = data['result']
print('OPEN_ITEMS:', r.get('openItemCount', 0))
print('AI_FINDINGS:', r.get('aiFindingCount', 0))
print('---ACTION_ITEMS---')
for item in r.get('actionItems', []):
    print('-', item)
print('---FEEDBACK---')
print(r.get('actionableFeedback', 'None'))
"
```

**Pass criteria:** `openItemCount: 0` (no unresolved critical/high items)
**On failure:** Fix issues using the `actionableFeedback` and `actionItems`. Commit, push, and re-run this gate.

### Gate 3: Merge Readiness

```bash
curl -s -X POST "${APP_URL}/api/review/merge-readiness" \
  -H "Content-Type: application/json" \
  -H "x-internal-api-secret: ${API_SECRET}" \
  -d '{"repo":"REPO","branch":"BRANCH","prNumber":PR_NUM}'
```

Parse with python3:
```bash
python3 -c "
import sys, json
data = json.load(sys.stdin)
if not data.get('success'):
    print('ERROR:', data.get('error', 'Unknown error'))
    sys.exit(1)
r = data['result']
print('IS_READY:', r.get('isReady', False))
print('AI_VERDICT_READY:', r.get('aiVerdictReady', False))
print('---BLOCKING---')
for issue in r.get('blockingIssues', []):
    print('-', issue)
print('---WARNINGS---')
for w in r.get('warnings', []):
    print('-', w)
"
```

**Pass criteria:** `aiVerdictReady: true`
**On failure:** Fix issues using the `blockingIssues`. Commit, push, and re-run the failing gate.

### Handling Findings — Including Pre-Existing Issues

The review APIs analyze the full branch, not just your diff. This means they may surface **pre-existing issues** in code you didn't write. This is intentional — the codebase should improve over time with every PR.

**Triage each finding into one of three categories:**

1. **Genuine issue (your code or pre-existing)** — Fix it. Real bugs, security issues, missing error handling, and code quality problems should be fixed regardless of whether you introduced them. Leave the codebase better than you found it.

2. **Intentional design choice / acceptable trade-off** — The code is correct but the AI disagrees with the pattern. For example: a deliberate fallback to env vars, a purposely loose type, or a known limitation documented in comments. These are not actionable.

3. **AI hallucination / stale data** — The finding references code that doesn't exist, misreads the logic, or flags a review comment that was already resolved. These are false positives.

**For categories 2 and 3:**
- Track them across runs. If the same non-actionable findings persist for **3 consecutive runs** after you've inspected the code and confirmed they don't warrant changes:
  - **Stop the review loop.**
  - Tell the user exactly which findings are persisting, your assessment of each (why it's intentional or a false positive), and ask for guidance on whether to proceed to finalization or make additional changes.
- **Never make unnecessary code changes** just to appease a finding you believe is wrong. Unnecessary changes introduce risk and noise in the PR.

### After fixing any gate

1. Commit and push the fixes:
   ```bash
   git add <specific-files-you-changed>
   git commit -m "fix: address review feedback - <brief description>"
   git push
   ```

2. Re-run the **failing gate** (not all gates — only repeat from the gate that failed).

3. Once a gate passes, move to the next gate.

4. If all 3 gates pass, exit the review loop.

5. Keep iterating until all 3 gates pass. You decide when the code is ready — there is no artificial limit. The only exception is persistent false positives (see above), where you should escalate to the user.

## Phase 5: Finalize & Notify

1. Mark the PR as ready for review (remove draft status):
   ```bash
   gh pr ready
   ```

2. Send a Slack notification:
   ```bash
   PR_URL=$(gh pr view --json url -q .url)
   curl -s -X POST "${APP_URL}/api/review/notify" \
     -H "Content-Type: application/json" \
     -H "x-internal-api-secret: ${API_SECRET}" \
     -d "{\"type\":\"ready_for_review\",\"repo\":\"REPO\",\"branch\":\"BRANCH\",\"data\":{\"prUrl\":\"${PR_URL}\",\"missionGoal\":\"MISSION_GOAL\"}}"
   ```

3. Tell the user:
   > All review gates passed. PR is ready for human review.
   > **PR:** [PR_URL]
   > **Branch:** BRANCH
   > **Slack notification sent.**

## Available Review Tools Reference

### 1. Code Quality (`POST /api/review/code-quality`)
AI-powered code quality analysis. Does NOT require a PR — reviews the branch directly.

**Request:** `{"repo": "owner/repo", "branch": "feat/my-feature", "missionGoal": "what the code should accomplish"}`

**Key response fields:**
- `result.passed` (boolean) — whether quality meets the bar
- `result.overallAssessment` — "healthy", "needs-attention", or "critical"
- `result.findings` (string) — detailed findings markdown
- `result.agentInstructions` (string) — specific fix instructions for agents

### 2. PR Review (`POST /api/review/pr-review`)
AI-powered analysis of PR review comments. Checks for unresolved or critical feedback from reviewers.

**Request:** `{"repo": "owner/repo", "prNumber": 42, "missionGoal": "what the code should accomplish"}`

**Key response fields:**
- `result.openItemCount` (number) — count of unresolved critical/high items
- `result.aiFindingCount` (number) — count of AI-generated findings
- `result.actionItems` (string[]) — list of specific actions to take
- `result.actionableFeedback` (string) — formatted feedback markdown

### 3. Merge Readiness (`POST /api/review/merge-readiness`)
Comprehensive merge readiness check. Requires a PR. Analyzes diff, reviews, docs, and build status.

**Request:** `{"repo": "owner/repo", "branch": "feat/my-feature", "prNumber": 42}`

**Key response fields:**
- `result.isReady` (boolean) — overall readiness
- `result.aiVerdictReady` (boolean) — AI's merge verdict
- `result.blockingIssues` (string[]) — issues that must be fixed
- `result.warnings` (string[]) — non-blocking warnings

### 4. Notify (`POST /api/review/notify`)
Send a Slack notification. Use after all checks pass.

**Request:** `{"type": "ready_for_review", "repo": "owner/repo", "branch": "feat/my-feature", "data": {"prUrl": "https://...", "missionGoal": "..."}}`

## Important Notes

- **Do NOT use `jq`** — it may not be available. Use `python3 -c "import sys,json; ..."` for JSON parsing.
- **Draft PR opened early** so PR-dependent checks (pr-review, merge-readiness) work from the start.
- **Self-assess before API calls** to avoid burning expensive AI calls on incomplete work.
- **No iteration limit** — keep going until all gates pass. You decide when the code is ready for human review.
- **Each review call takes 30-90 seconds** — tell the user to expect a wait.
- **Push before re-reviewing** — the review APIs check the latest commit on the branch.
- **Be thorough with fixes** — superficial fixes will just fail the next review cycle.
- **All auth headers** use `x-internal-api-secret` with the resolved `API_SECRET` value.
