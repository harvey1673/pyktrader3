# AI Skill Convention

## Overview

This repository maintains an **AI skill library** under `docs/skills/`. Each skill file documents a known capability in a standardised format so that AI assistants can reuse established patterns instead of rediscovering or reimplementing them.

When an AI agent learns something new about this codebase — for example, how data is loaded, how signals are generated, or how backtests are structured — that knowledge should be captured as a skill file. Future agents (and human developers) can read the skill file and immediately apply the capability correctly.

---

## Skill File Location

All skill files are stored under:

```
docs/skills/
```

File names should be lowercase with underscores, e.g.:

```
docs/skills/data_loading.md
docs/skills/signal_generation.md
docs/skills/backtest_execution.md
```

---

## Required Skill Structure

Every skill file must contain the following sections in order:

| Section | Purpose |
|---|---|
| **Skill Name** | Short canonical name |
| **Purpose** | One-paragraph summary of what the skill does |
| **When To Use** | Conditions that indicate this skill is applicable |
| **Functions Used** | List of repository functions with file paths and one-line descriptions |
| **Inputs** | Parameters, types, and expected formats |
| **Outputs** | Return types and DataFrame structures |
| **Example Usage** | Working code snippets that can be run directly |
| **Implementation Notes** | Conventions, edge cases, known caveats |
| **Common Patterns** | Short reusable code blocks for typical workflows |
| **Related Skills** | Links to other skills in `docs/skills/` |

---

## Skill Writing Guidelines

- **Reference existing functions** rather than reimplementing logic. Always prefer calling `load_hist_fut_prices()`, `nearby()`, etc. over writing equivalent logic inline.
- **Prefer small examples** over long explanations. A 5-line code block teaches more than three paragraphs of prose.
- **Document data formats explicitly**. State whether a DataFrame uses a DatetimeIndex, what MultiIndex levels look like, what column names to expect.
- **Write for autonomous execution**. Assume the reader is an AI agent with no prior context. The skill file must be self-contained enough that the agent can produce working code after reading it.
- **Keep each skill focused**. One skill = one capability. Prefer multiple small skill files over one large file.
- **Include failure modes**. If a function can silently return an empty DataFrame or raise a known exception, document that.

---

## Example Skill Skeleton

````markdown
# Skill: <Skill Name>

## Purpose
<One paragraph describing what this skill does.>

## When To Use
- <Situation 1>
- <Situation 2>

## Functions Used

| Function | File | Description |
|---|---|---|
| `function_name()` | `path/to/file.py` | What it does |

## Inputs

| Parameter | Type | Description |
|---|---|---|
| `param1` | `str` | Description |
| `param2` | `datetime.date` | Description |

## Outputs

| Name | Type | Structure |
|---|---|---|
| `result` | `pd.DataFrame` | DatetimeIndex, columns = [...] |

## Example Usage

```python
# Example description
from module import function_name

result = function_name(param1='value', param2=datetime.date(2025, 1, 1))
```

## Implementation Notes
- Note 1
- Note 2

## Common Patterns

```python
# Pattern description
```

## Related Skills
- [Other Skill](other_skill.md)
````
