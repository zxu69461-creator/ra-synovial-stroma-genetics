# CLAUDE.md

Instructions for Claude Code in this repository.

## Git identity and attribution

These rules apply to Claude Code (terminal, desktop, web, mobile, cloud sessions) working in this repository on behalf of its owner. They were set by the owner and must not be relaxed without an explicit instruction in the current conversation.

- You are a coding assistant, not a Git author. Never set yourself (for example `claude <noreply@anthropic.com>`) as author, committer or co-author.
- Commits made for the owner carry the owner's verified GitHub identity: `Zhida Xu <265443646+zxu69461-creator@users.noreply.github.com>` (GitHub account `zxu69461-creator`). Apply this only when you are working for the owner; confirm that with `gh api user --jq .login` or the GitHub connection of the current session. If you cannot confirm who you are acting for, stop before committing and report it instead of guessing. Never put the owner's name on a commit written by someone else.
- Cloud machines are temporary, so repeat the check in every new session and before creating a branch, committing, pushing or opening a pull request: read `git var GIT_AUTHOR_IDENT GIT_COMMITTER_IDENT`, make sure `GIT_AUTHOR_*` / `GIT_COMMITTER_*` environment variables are not overriding the configuration, and if the machine has another identity set `user.name` and `user.email` for this clone only (`git config`, not `--global`). If the identity still cannot be confirmed, stop and report why.
- Commit messages and pull request descriptions must not contain `Co-Authored-By` lines for Claude, a "Generated with Claude Code" footer, a `Claude-Session` line or a claude.ai session link. `.claude/settings.json` keeps `attribution` switched off (`commit` and `pr` empty, `sessionUrl` false) on purpose: do not remove it, do not turn it back on, and do not rely on the deprecated `includeCoAuthoredBy`. If user, project or managed settings conflict with it, report the conflict.
- Before opening a pull request, confirm the GitHub account in use is the owner's. The pull request creator is a different thing from the commit author and committer; never create pull requests with a bot account.
- Never remove or change the identity of real third-party contributors, and never strip copyright, licence or credit lines. Do not rewrite published history, force-push, or change protected branches without the owner's explicit approval; before rewriting an unmerged branch, back it up, record the old and new SHAs, confirm the file tree is unchanged, and ask first.
- Safeguards in this repository: `.githooks/pre-commit` and `.githooks/commit-msg` reject Claude or hostname-fallback identities and automatic attribution (a session hook in `.claude/settings.json` enables them with `git config core.hooksPath .githooks`; terminal users can run that command once), and the `Commit identity check` workflow checks the owner's pull requests. Local hooks do not protect edits made in the GitHub web UI or on machines where they are not enabled.
- Report configuration conflicts; never bypass a check silently.
