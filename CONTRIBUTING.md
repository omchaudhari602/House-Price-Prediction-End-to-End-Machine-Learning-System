# Contributing Guide

Thanks for your interest in contributing to this project.

## Development Workflow

1. Create a feature branch from your local default branch.
2. Make focused, minimal changes.
3. Run tests before submitting changes.
4. Update docs if behavior or interfaces changed.
5. Open a pull request with clear summary and rationale.

## Code Standards

- Keep modules single-purpose and readable.
- Add docstrings for public functions.
- Prefer explicit error messages over silent failures.
- Preserve existing project structure unless refactor is necessary.

## Testing Requirements

Before submitting:

- Ensure test suite passes (`pytest`).
- Add/adjust tests for new behavior.
- Avoid introducing flaky tests.

## Documentation Requirements

If your change affects setup, API behavior, or ML workflow, update:

- `README.md`
- relevant files in `docs/`
- screenshot references in `docs/screenshots/README.md` (if UI/API output changed)

## Commit Message Suggestions

Use concise, scoped messages. Examples:

- `docs: add training workflow details`
- `fix: handle missing fields in predict payload`
- `test: add inference schema validation test`

## Pull Request Checklist

- [ ] Tests pass locally
- [ ] Docs updated (if needed)
- [ ] No unrelated file changes
- [ ] Clear PR description with impact and verification
