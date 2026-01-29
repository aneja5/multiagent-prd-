# Multi-Agent PRD Generator

A sophisticated tool that generates research-backed Product Requirements Documents (PRDs) using multiple AI agents coordinating through a shared state object. Built with the ReAct (Reasoning + Acting) framework.

## Overview

This tool transforms a simple product idea into a comprehensive PRD by:
- Clarifying ambiguous requirements through intelligent questioning
- Conducting web research to gather evidence and insights
- Analyzing competitors, pain points, and user workflows
- Synthesizing findings into a well-structured PRD with citations

## Features

- **Multi-Agent Architecture**: Specialized agents work together to handle different aspects of PRD generation
- **ReAct Framework**: Each agent uses a Think-Act-Observe-Update-Reflect loop for intelligent decision-making
- **Research-Backed**: All claims in the PRD are backed by web research with proper citations
- **Stateful & Resumable**: Complete execution state is persisted, allowing runs to be paused and resumed
- **Production-Ready**: Comprehensive error handling, logging, retry logic, and type safety
- **Rich CLI**: Beautiful command-line interface with progress tracking and formatted output

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLI (main.py)                       │
│  - Parse arguments                                          │
│  - Initialize orchestrator                                  │
│  - Display results                                          │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator                             │
│  - Coordinate agent execution                               │
│  - Manage workflow                                          │
│  - Determine agent selection                                │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  Agent   │   │  Agent   │   │  Agent   │
    │    1     │   │    2     │   │    N     │
    └────┬─────┘   └────┬─────┘   └────┬─────┘
         │              │              │
         └──────────────┼──────────────┘
                        ▼
              ┌──────────────────┐
              │   Shared State   │
              │                  │
              │  - Metadata      │
              │  - Research Plan │
              │  - Evidence      │
              │  - Insights      │
              │  - PRD           │
              │  - Task Board    │
              │  - Agent Trace   │
              └──────────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  Persistent      │
              │  Storage         │
              │  (JSON files)    │
              └──────────────────┘
```

## Project Structure

```
multiagent-prd/
├── app/
│   ├── __init__.py
│   ├── main.py              # CLI interface
│   ├── config.py            # Configuration management
│   ├── logger.py            # Logging setup
│   ├── state.py             # State schema and persistence
│   └── orchestrator.py      # Agent coordination
├── agents/
│   ├── __init__.py
│   ├── base_agent.py        # Base agent with ReAct framework
│   └── prompts/             # Prompt templates (created as needed)
├── data/
│   ├── runs/                # Saved run states (auto-created)
│   └── logs/                # Application logs (auto-created)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

### Prerequisites

- Python 3.9+
- OpenAI API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd multiagent-prd
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-2024-08-06
LOG_LEVEL=INFO
```

## Usage

### Start a New PRD Generation

```bash
python -m app.main "Build a project management tool for remote teams"
```

### Resume an Existing Run

```bash
python -m app.main --resume <run-id>
```

### List All Runs

```bash
python -m app.main --list
```

## State Schema

The system uses a comprehensive state schema that tracks all aspects of PRD generation:

```python
State {
    run_id: str                    # Unique run identifier
    created_at: str                # ISO timestamp
    status: "running|blocked|done" # Current status

    metadata: {
        raw_idea: str              # Original product idea
        domain: str                # Product domain
        industry_tags: [str]       # Industry classifications
        target_user: str           # Target audience
        geography: str             # Geographic focus
        compliance_contexts: [str] # Regulatory requirements
        prd_style: str            # Output format preference
        clarification_status: str  # Clarification state
    }

    research_plan: {
        queries: [Query]           # Research queries to execute
    }

    evidence: [Evidence]           # Collected research evidence
    insights: {
        pain_points: [PainPoint]   # Identified pain points
        competitors: [Competitor]  # Competitor analysis
        workflows: [Workflow]      # User workflows
    }

    prd: {
        sections: {}               # PRD content sections
        notion_markdown: str       # Formatted output
        citation_map: {}           # Evidence citations
    }

    task_board: [Task]            # Agent task management
    agent_trace: [AgentTraceEntry] # Execution history
}
```

## Agent Development

### Creating a New Agent

1. Create a new file in `agents/`:
```python
from agents.base_agent import BaseAgent
from app.state import State

class MyAgent(BaseAgent):
    def run(self, state: State) -> State:
        # 1. Think: Analyze state
        analysis = self._think(state)

        if not analysis["should_act"]:
            return state

        # 2. Act: Call LLM
        prompt = self._load_prompt()
        messages = [{"role": "user", "content": prompt}]
        response = self._call_llm(messages)

        # 3. Observe: Parse response
        observations = self._observe(response)

        # 4. Update: Modify state
        state = self._update_state(state, observations)

        # 5. Reflect: Log action
        self._log_action(state, "Completed my task")

        return state
```

2. Create a prompt template in `agents/prompts/my_agent.txt`

3. Register the agent in `app/main.py`:
```python
from agents.my_agent import MyAgent

orchestrator.register_agent(MyAgent("my_agent", client))
```

## Development Roadmap

### Phase 1: Foundation ✅
- [x] State schema and persistence
- [x] Configuration management
- [x] Logging infrastructure
- [x] Base agent with ReAct framework
- [x] CLI interface
- [x] Orchestrator skeleton

### Phase 2: Core Agents (Next)
- [ ] Clarification Agent
- [ ] Research Planner Agent
- [ ] Web Search Tool Integration
- [ ] Research Execution Agent
- [ ] Insight Synthesis Agent

### Phase 3: PRD Generation
- [ ] PRD Writer Agent
- [ ] Citation Manager
- [ ] Quality Review Agent
- [ ] Notion Markdown Formatter

### Phase 4: Enhancements
- [ ] Parallel agent execution
- [ ] Advanced orchestration logic
- [ ] User interaction during execution
- [ ] Web UI
- [ ] Export formats (PDF, HTML)

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key (required) | - |
| `OPENAI_MODEL` | Model to use | `gpt-4o-2024-08-06` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MAX_RETRIES` | API retry attempts | `3` |
| `RETRY_DELAY` | Delay between retries (seconds) | `1` |
| `OUTPUT_DIR` | Directory for run data | `data/runs` |
| `LOG_DIR` | Directory for logs | `data/logs` |

## Error Handling

The system includes comprehensive error handling:

- **API Errors**: Automatic retry with exponential backoff
- **Configuration Errors**: Clear error messages with resolution hints
- **State Persistence**: Atomic writes with validation
- **Agent Errors**: Logged and traced for debugging

## Logging

Logs are written to both console (with rich formatting) and file:

- Console: Colored output with timestamps
- File: `data/logs/app.log` with detailed information

## Testing

```bash
# Run tests (when implemented)
pytest

# Type checking
mypy app/ agents/

# Code formatting
black app/ agents/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper type hints and docstrings
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

For issues and questions:
- Open an issue on GitHub
- Check the documentation
- Review the agent trace logs for debugging

## Acknowledgments

Built with:
- OpenAI GPT-4
- Pydantic for data validation
- Rich for beautiful CLI output
- ReAct framework for agent reasoning
