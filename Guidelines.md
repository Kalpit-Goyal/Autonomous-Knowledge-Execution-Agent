
Objective
Build an autonomous AI agent that can perceive its environment, reason about goals, and take independent action to achieve specific outcomes with minimal human supervision.

Submission Requirements
Please submit the following:

Source Code

README

Installation Steps

GitHub Repository link or ZIP file

General Instructions
Do not use hardcoded responses, predefined rules, or static outputs. Your solution should rely on autonomous reasoning using an LLM.

Do not copy code from public repositories or other external sources. Your submission should primarily represent your own work. If we determine that your solution has been copied or significantly derived from publicly available repositories, your submission may be disqualified.

You may use any programming language, AI framework, libraries, or tools of your choice (e.g., LangGraph, OpenAI Agents SDK, CrewAI, AutoGen, MCP, vector databases, etc.), wherever appropriate.

Ensure your README contains clear setup instructions so the solution can be executed without additional guidance.

Assessment Option 3: Autonomous Knowledge Execution Agent
Goal: Build an autonomous AI agent capable of retrieving information from
internal knowledge sources, reasoning over the retrieved data, making
intelligent decisions, and executing appropriate actions based on user
requests.

Requirements:
 Use one or more internal knowledge sources (Database, JSON, CSV,
Vector Database, etc.).
 Answer user queries using only the available internal knowledge.
 Reason over the retrieved information before making decisions.
 Determine the appropriate action based on the user&#39;s request and the
available knowledge.
 Execute the selected action automatically.
 Explain why the selected action was performed.

Bonus:
 Add long-term memory.
 Execute multiple actions in parallel where applicable.
 Support multi-step reasoning and planning.
 Require human approval before executing critical or irreversible actions.
 Maintain audit logs for every action performed.
 Support multiple knowledge sources simultaneously.
 Handle incomplete or conflicting information gracefully.
 Provide reasoning for each decision before execution.