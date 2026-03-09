# Project Leapfrog Submission

## 1. Problem Statement

### What problem did you choose?
For Project Leapfrog, I chose to build an open-source market-research and competitive-analysis workflow around local business discovery. The project started from a concrete technical limitation: Google Places API results are capped, which means naive queries can miss a meaningful portion of businesses in dense areas. I built a Python-based toolkit to improve coverage through adaptive geographic subdivision, deduplication, and structured data collection, and then extended it toward a broader AI-assisted analysis workflow.

### Why does this problem matter?
This problem matters because market discovery and competitive research are useful inputs for business planning and strategy, but the workflow is often incomplete, fragmented, and manual. Solving the raw data-collection problem is only the first step. The larger opportunity is turning incomplete external data into structured insights through a repeatable engineering workflow.

## 2. Initial Approach (Before AI)

### How would you have approached this problem traditionally?
Without AI, I would have approached this as a more sequential engineering project. I would first focus on the API and data-collection layer, solve result-limit and deduplication issues, add tests, and only then move to downstream analytics or reporting. I also would have been more conservative in scope because broadening the project would require much more manual design, implementation, and validation effort.

### What steps, roles, or time would this normally require?
Traditionally, this kind of work would require me to switch manually across multiple roles: engineer for implementation, QA for testing strategy, reviewer for correctness, and product/analyst for output design. All of the decomposition, scaffolding, code review, and iteration loops would be slower. Expanding from a focused toolkit into a broader analysis workflow would take substantially more elapsed time and coordination.

## 3. AI-First Approach

### What AI tools or environments did you use?
I used AI coding assistants in a CLI-driven workflow and intentionally compared different models for implementation, debugging, testing, and review. One of my clearest takeaways is that models behave very differently and should not be treated as interchangeable. Different models were better suited for different tasks, and in some cases one model produced more verbose code than I would prefer, sometimes more verbose than both human code and other models. I also used AI-generated code review, AI-assisted test creation, design brainstorming tools, and structured artifacts such as GitHub issues, implementation plans, and PRs as sources of truth.

I also found value in using multiple review surfaces. Asking different tools and models to review the same PR was often effective at finding different classes of issues. In practice, requesting Codex 5.3, Augment, Cursor bot, and Claude-based reviewers to look at the same PR often helped surface issues from different angles. In contrast, I found the IDE itself less central than I expected. In many cases, comparing diffs carefully and driving work through the CLI plus PR-based review was already good enough.

### How did AI influence problem framing, design, or execution?
AI changed both the pace and the shape of the work. It made it much easier for me to decompose the project into smaller stages, issues, and reviewable units. It also lowered the cost of exploring ideas, so I was able to expand from a narrower coding task into a broader workflow system.

At the same time, I found that AI only worked well when I constrained it with explicit sources of truth: actual code, implementation plans, GitHub issues, tests, and acceptance criteria. AI was useful for brainstorming and accelerating implementation, but I learned quickly that it should not be trusted to define reality on its own.

One practical lesson for me was that some tools are useful for generating an initial design or AI-friendly plan for a medium-sized task, but I should be very careful about relying on one-shot parallel implementation for anything non-trivial. Better outcomes came from breaking work into labeled issues, understanding dependencies, using PRs consistently, and validating incrementally.

Claude Plugins, such as `superpowers:brainstorm`, were useful for producing an initial design for medium-sized tasks and helping me come up with AI-friendly plans. However, I would be careful not to rely on plugin-driven parallel implementation to achieve non-trivial features in one shot. Project management best practices still mattered a lot: creating issues or tickets with labels, doing dependency analysis, always creating PRs, and maintaining a strong CI/CD pipeline with both unit tests and integration tests for each PR all helped guardrail project quality.

## 4. What Changed

### What felt faster or easier?
The biggest benefit I felt was speed of innovation. AI made it much easier to quickly test ideas, build small proofs of concept, validate approaches, and iterate on designs. This is probably the most exciting and valuable benefit I see today. For new ideas, greenfield projects, and quick experiments, AI can dramatically reduce the cost of trying something.

AI also made several engineering tasks faster:

- exploring multiple design paths
- generating scaffolding and repetitive code
- creating first-pass tests and review comments
- iterating quickly across smaller tasks
- moving between implementation, critique, and design discussion

This experience made me more convinced that we should encourage engineers to use AI heavily for quick POCs, fast validation of ideas, and rapid iteration in early-stage exploration.

### What felt harder or more confusing?
The hardest part was not generating code. It was deciding what to trust.

Several recurring failure modes stood out for me:

- models behave very differently on planning, coding, verbosity, debugging, and test-heavy tasks
- some models produce more verbose code than I would prefer, sometimes more verbose than both human code and other models
- AI reviewers can confidently make incorrect claims
- advanced models may admit my comments too quickly, especially when I use words like “critical,” “important,” or “severe,” without actually double-checking the design or implementation facts
- AI-generated tests are not automatically trustworthy
- coding agents may try to dismiss failures as unrelated, flaky, or due to non-determinism
- models often produce plausible post-hoc explanations for mistakes instead of evidence-based explanations

I also became much more cautious when using AI on an existing codebase. For greenfield work, AI can be a huge accelerator. For a real codebase with history, assumptions, edge cases, and testing expectations, AI can still be very useful, but it requires much more discipline and skepticism. I learned that “unrelated” is a hypothesis that needs to be validated, not a conclusion I should accept automatically, and I should not let the coding agent waive suspicious failures just because it claims they are unrelated.

### What surprised you?
What surprised me most was how often AI looked productive while bypassing the most important engineering behavior: verification.

The models could generate code, tests, documentation updates, and review responses very quickly, but they often needed to be pushed back to the source of truth. They might trust a reviewer comment without reading the code, accept my framing without confirming facts, or generate tests that preserved a fix without actually detecting the bug proactively. In some cases, the model seemed to optimize for pleasing me rather than independently validating facts, especially when the feedback was framed as urgent or severe.

Another surprise was that if I challenged the model one more time before merging a PR, I often found more issues. That reinforced for me that the human role is not just to prompt, but to interrogate.

I was also surprised by the economics. AI coding looks efficient on the surface, but the actual token consumption can become very large when I am using heavyweight models, long contexts, repeated review cycles, and lots of testing/debugging.

## 5. Impact on Engineering Work

### How does this experience change your view of engineering roles?
This experience strengthened my view that engineering roles are shifting more toward decomposition, validation, orchestration, and judgment. The scarce skill is becoming less about writing every line manually and more about defining the right task boundaries, establishing the source of truth, challenging weak reasoning, and building reliable review loops.

It also made me appreciate even more that strong software and system experience matters. In some sense, it matters more than before. Experienced engineers are better at producing clear problem statements, reviewing designs, spotting weak assumptions, defining good guardrails, and diagnosing issues when the AI output looks plausible but is actually wrong.

In an AI-first workflow, engineers increasingly need to:

- break ambiguous work into clear, testable units
- verify claims against code and artifacts
- evaluate whether tests actually detect bugs
- review outputs across multiple passes and sometimes multiple models
- manage integration confidence, not just local correctness

### What new skills or behaviors seem more important?
Several skills now feel more important to me:

- verification discipline over trusting fluent explanations
- strong test design, especially bug-detection tests
- adversarial review behavior
- source-of-truth orientation using code, issues, and implementation plans
- model-selection judgment
- knowing when to use multiple reviewers or models
- distinguishing regression preservation from true bug discovery
- refusing to waive suspicious failures without proof
- designing AI-friendly plans rather than relying on one-shot generation

One of my clearest conclusions is that if the goal is production-ready code, test-driven development with AI is not optional. It is mandatory.

## 6. Leadership Reflections

### How should leaders adapt expectations, processes, or support?
My biggest leadership takeaway is that leaders should encourage AI-driven innovation aggressively, but enforce software development discipline even more strongly.

On the positive side, AI is a very strong tool for innovation speed. It lowers the cost of trying ideas, creating quick POCs, and validating directions early. I think this is one of the best uses of AI today. For new ideas, new projects, and early exploration, we should actively encourage engineers to use AI and iterate faster. If AI helps us validate ten ideas instead of two, that is a real advantage.

At the same time, when AI is used against an existing codebase or when production-quality code is the goal, leaders should enforce best software development discipline through tools and process guardrails. That means:

- enforce PR-based workflows
- enforce testing at every change
- enforce code reviews
- enforce CI/CD quality gates
- enforce both unit tests and integration tests where applicable
- enforce explicit issue tracking, acceptance criteria, and dependency awareness
- enforce quality control rather than accepting AI output at face value

In other words, AI should not reduce engineering discipline. It should increase the need for discipline, and we should use tools to enforce it.

Leaders should also separate speed of output from confidence in quality. AI can create a lot of visible activity very quickly, but that does not mean the code is correct, maintainable, or production-ready.

Another practical leadership lesson is that different models are better suited for different tasks. Teams should be encouraged to learn those differences instead of assuming one model fits everything. Model choice, review workflow, and engineering guardrails should become part of normal team practice.

Finally, leaders should recognize two operational realities:  
First, AI increases workload in some ways because model throughput is much higher than human review throughput. Even if reviewers only inspect the critical parts, review bandwidth can become a bottleneck.  
Second, AI coding is not automatically cost-effective. Faster generation does not necessarily mean lower engineering cost.

To make that concrete, I tracked one day of usage and saw the following pattern:

| Hour (PST) | Input Tokens | Output Tokens | API Calls |
|---|---:|---:|---:|
| 9:00 | 12.8M | 35K | 163 |
| 10:00 | 23.6M | 40K | 222 |
| 11:00 | 32.5M | 61K | 334 |
| 12:00 | 8.0M | 12K | 92 |
| 13:00–14:00 | — | — | — |
| 15:00 | 11.2M | 20K | 132 |
| 16:00 | 22.8M | 21K | 165 |

- Peak hour: 11:00 AM with 32.5M input tokens and 334 API calls during heavy PR comment addressing and test writing
- Lunch break gap: 1–3 PM with zero activity
- Afternoon ramp-up: 3–5 PM for integration tests and PR creation
- Total: about 110.8M input tokens, 193K output tokens, and 1,113 API calls
- Estimated cost: about $1,677 at Opus pricing, though actual billing would be lower because a meaningful portion of input tokens were cached

That example does not mean AI is not valuable. It does mean we should be more thoughtful about where AI creates the most leverage. My current view is that AI is most compelling for rapid experimentation, quick validation, and acceleration of well-scoped tasks. Its cost-effectiveness is less obvious when large models are used heavily for repeated coding, review, and debugging loops.

### What might teams need more or less of during this transition?
Teams likely need more:

- quick POCs and fast validation cycles for new ideas
- smaller and clearer GitHub issues
- AI-friendly design and implementation plans
- mandatory PR workflows
- stronger TDD expectations
- CI/CD with both unit and integration tests on every PR
- multiple review mechanisms where appropriate
- coaching on how to challenge AI outputs effectively

Teams likely need less:

- blind trust in AI-generated review comments
- acceptance of “unrelated” failures without investigation
- one-shot implementation for non-trivial work
- superficial test counts as a proxy for quality
- the assumption that faster code generation means less engineering discipline

My personal conclusion is that we should push teams to use AI more aggressively for innovation and idea validation, but we should also tighten—not relax—our expectations around testing, reviews, and quality control.

## 7. Open Questions

### What remains unclear?
Several questions remain open for me:

- What is the right balance between AI speed and engineering confidence?
- How should teams systematically choose the right model for the right task?
- How should review norms evolve when both the code and the reviews may be AI-assisted?
- What is the best way to distinguish true non-determinism or flakiness from genuine implementation defects?
- How do we control review workload when AI throughput exceeds human verification capacity?
- Under what conditions is AI coding actually cost-effective after factoring in tokens, retries, and review effort?

### What would you want to explore next?
Next, I want to explore:

- a more systematic multi-model workflow for implementation, critique, and review
- stronger prompting patterns for adversarial testing and bug discovery
- how GitHub issues, labels, dependency analysis, and implementation plans can serve as stronger control surfaces for AI coding
- better ways to measure productivity that include verification cost, not just coding speed
- which parts of engineering benefit most from AI and which still depend primarily on human systems experience and judgment
