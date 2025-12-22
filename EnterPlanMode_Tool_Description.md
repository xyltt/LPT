# EnterPlanMode Tool

## Overview

The **EnterPlanMode** tool is used when encountering a complex task that requires careful planning and exploration before implementation. This tool transitions the assistant into plan mode where it can thoroughly explore the codebase and design an implementation approach.

## Purpose

Use this tool to:
- Thoroughly explore the codebase before making changes
- Design implementation approaches for complex tasks
- Present detailed plans for user approval
- Make informed architectural decisions

## When to Use EnterPlanMode

Use this tool when **ANY** of these conditions apply:

### 1. Multiple Valid Approaches
The task can be solved in several different ways, each with trade-offs.

**Examples:**
- "Add caching to the API" - could use Redis, in-memory, file-based, etc.
- "Improve performance" - many optimization strategies possible

### 2. Significant Architectural Decisions
The task requires choosing between architectural patterns.

**Examples:**
- "Add real-time updates" - WebSockets vs SSE vs polling
- "Implement state management" - Redux vs Context vs custom solution

### 3. Large-Scale Changes
The task touches many files or systems.

**Examples:**
- "Refactor the authentication system"
- "Migrate from REST to GraphQL"

### 4. Unclear Requirements
You need to explore before understanding the full scope.

**Examples:**
- "Make the app faster" - need to profile and identify bottlenecks
- "Fix the bug in checkout" - need to investigate root cause

### 5. User Input Needed
You'll need to ask clarifying questions before starting.

**Note:** If you would use AskUserQuestion to clarify the approach, consider EnterPlanMode instead. Plan mode lets you explore first, then present options with context.

## When NOT to Use EnterPlanMode

Do **NOT** use EnterPlanMode for:
- Simple, straightforward tasks with obvious implementation
- Small bug fixes where the solution is clear
- Adding a single function or small feature
- Tasks you're already confident how to implement
- Research-only tasks (use the Task tool with explore agent instead)

## What Happens in Plan Mode

When in plan mode, the assistant will:

1. **Explore** - Thoroughly explore the codebase using Glob, Grep, and Read tools
2. **Understand** - Understand existing patterns and architecture
3. **Design** - Design an implementation approach
4. **Present** - Present the plan to the user for approval
5. **Clarify** - Use AskUserQuestion if needed to clarify approaches
6. **Exit** - Exit plan mode with ExitPlanMode when ready to implement

## Examples

### ✅ GOOD - Use EnterPlanMode

**Example 1:**
```
User: "Add user authentication to the app"
```
- Requires architectural decisions (session vs JWT, where to store tokens, middleware structure)

**Example 2:**
```
User: "Optimize the database queries"
```
- Multiple approaches possible, need to profile first, significant impact

**Example 3:**
```
User: "Implement dark mode"
```
- Architectural decision on theme system, affects many components

### ❌ BAD - Don't use EnterPlanMode

**Example 1:**
```
User: "Fix the typo in the README"
```
- Straightforward, no planning needed

**Example 2:**
```
User: "Add a console.log to debug this function"
```
- Simple, obvious implementation

**Example 3:**
```
User: "What files handle routing?"
```
- Research task, not implementation planning

## Important Notes

- **User Approval Required** - This tool requires user consent before entering plan mode
- **Be Thoughtful** - Unnecessary plan mode slows down simple tasks
- **When in Doubt** - If unsure whether to use it, err on the side of starting implementation
- **Alternative** - You can always ask the user "Would you like me to plan this out first?"

## Tool Parameters

This tool takes no parameters. Simply invoke it when the conditions for planning are met.

## Related Tools

- **ExitPlanMode** - Exit plan mode when planning is complete and ready to implement
- **Task (Plan agent)** - Launch a specialized planning agent for software architecture
- **TodoWrite** - Create structured task lists for planning
