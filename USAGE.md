# Use Issue2PR on your repo

Auto-review your pushed code: on every push, an AI agent reviews the changed
files, fixes bugs, opens a pull request, and emails whoever pushed.

## 5 steps

1. **Add the caller workflow.** In YOUR repo, create
   `.github/workflows/agent.yml` with:

   ```yaml
   name: auto-review on push
   on:
     push:
       branches-ignore: ['agent/**']   # don't react to the agent's own PRs
   jobs:
     agent:
       uses: <OWNER>/issue2pr/.github/workflows/agent.yml@main
       secrets: inherit
   ```
   Replace `<OWNER>` with the GitHub account/org that hosts issue2pr.

2. **Get a Groq API key** (free): https://console.groq.com → API Keys.

3. **(Optional) Get an OpenRouter key** for a free fallback model:
   https://openrouter.ai → Keys. Used only when Groq fails or is rate-limited.

4. **(Optional) Email reports.** Create a Gmail *App Password*
   (Google Account → Security → App passwords). Reports are emailed to whoever
   pushed. Skip this to get no email (the PR is still opened).

5. **Add the secrets** in YOUR repo → Settings → Secrets and variables →
   Actions → New repository secret:

   | Secret | Required | Value |
   |--------|----------|-------|
   | `GROQ_API_KEY` | yes | your Groq key |
   | `OPENROUTER_API_KEY` | no | your OpenRouter key (fallback) |
   | `MAIL_USERNAME` | no | your Gmail address (sender) |
   | `MAIL_PASSWORD` | no | your Gmail app password |

That's it. Push code → the agent runs → a PR appears → (optional) email lands.

## Notes

- **You bring your own keys.** Each repo uses its own secrets; nobody shares keys.
- **Skip one push:** put `[skip agent]` in the commit message.
- **Turn it off:** delete the caller file, or Actions tab → the workflow → Disable.
- **Forks:** GitHub does not give secrets to pushes from forks (security). The
  agent runs with keys only for pushes into the repo by collaborators.
- **Change the task:** pass an input in the caller, e.g.
  `with: { task: "Add unit tests for the changed functions." }`.
- **Private-email users:** if a pusher hid their email on GitHub, their commit
  email is a `@users.noreply.github.com` address and the report will bounce.
