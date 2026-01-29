# Multi-Agent PRD Generator

A sophisticated tool that generates research-backed Product Requirements Documents (PRDs) using multiple AI agents coordinating through a shared state object. Built with the ReAct (Reasoning + Acting) framework.

## 🎉 Day 2 Complete!

Two agents are live and the orchestrator is running! Here's what's working:

- ✅ **ClarificationAgent** - Extracts structured metadata from product ideas
- ✅ **PlannerAgent** - Generates 15-20 domain-specific research queries
- ✅ **DAG Orchestrator** - Manages task dependencies and agent sequencing
- ✅ **Multi-Domain Support** - Tested across fintech, healthcare, devtools, ecommerce, real estate
- ✅ **Test Suite** - 37 passing tests
- ⏳ **ResearcherAgent** - Coming in Day 3
- ⏳ **SynthesisAgent** - Coming in Day 4
- ⏳ **PRDWriterAgent** - Coming in Day 5

## Overview

This tool transforms a simple product idea into a comprehensive PRD by:
- ✅ **Clarifying ambiguous requirements** through intelligent metadata extraction (DONE)
- ✅ **Planning targeted research** with domain-specific queries and competitor analysis (DONE)
- ⏳ Conducting web research to gather evidence and insights (COMING SOON)
- ⏳ Analyzing competitors, pain points, and user workflows (COMING SOON)
- ⏳ Synthesizing findings into a well-structured PRD with citations (COMING SOON)

## Features

- **Multi-Agent Architecture**: Specialized agents work together to handle different aspects of PRD generation
- **ReAct Framework**: Each agent uses a Think-Act-Observe-Update-Reflect loop for intelligent decision-making
- **Research-Backed**: All claims in the PRD are backed by web research with proper citations
- **Stateful & Resumable**: Complete execution state is persisted, allowing runs to be paused and resumed
- **Production-Ready**: Comprehensive error handling, logging, retry logic, and type safety
- **Rich CLI**: Beautiful command-line interface with progress tracking and formatted output

## Current Architecture (Day 2)

```
User Input: "Build a HIPAA-compliant patient portal"
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    DAG Orchestrator                         │
│  - Task dependency resolution                               │
│  - Agent scheduling                                         │
│  - State checkpointing                                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌──────────────────────┐      ┌──────────────────────┐
│  ClarificationAgent  │      │    PlannerAgent      │
│  ✅ COMPLETE         │─────▶│    ✅ COMPLETE       │
│                      │      │                      │
│  Extracts:           │      │  Generates:          │
│  - domain            │      │  - 15-20 queries     │
│  - industry_tags     │      │  - 4 categories      │
│  - target_user       │      │  - priority levels   │
│  - geography         │      │  - expected sources  │
│  - compliance        │      │                      │
└──────────────────────┘      └──────────────────────┘
          │                               │
          └───────────────┬───────────────┘
                          ▼
                 ┌──────────────────┐
                 │   Shared State   │
                 │                  │
                 │  ✅ metadata     │
                 │  ✅ research_plan│
                 │  ⏳ evidence     │
                 │  ⏳ insights     │
                 │  ⏳ prd          │
                 └──────────────────┘
                          │
                          ▼
             Saved to: data/runs/{run_id}.json
```

## Full Architecture (When Complete)

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
    │ Clarify  │   │ Research │   │  PRD     │
    │  Agent   │   │  Agent   │   │  Writer  │
    │   ✅     │   │   ⏳     │   │   ⏳     │
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

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set up environment
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Run your first PRD generation
python -m app.main "Build a project management tool for remote teams"

# 4. See the extracted metadata
# Output will show a formatted table with domain, industry tags, target user, etc.
```

## Usage Examples

### Generate a New PRD

```bash
# Basic usage
python -m app.main "Build a HIPAA-compliant patient portal"

# With verbose output (shows agent trace)
python -m app.main "AI-powered scheduling assistant" --verbose

# Short form
python -m app.main "Invoice tracking for freelancers" -v
```

**Output:**
```
✓ Clarification Complete
                       Extracted Metadata
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Field                ┃ Value                                  ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Domain               │ healthcare                             │
│ Industry Tags        │ patient_engagement, EMR, telehealth    │
│ Target User          │ small medical clinics (2-10 providers) │
│ Geography            │ US                                     │
│ Compliance           │ HIPAA, state_medical_boards            │
│ Status               │ pending                                │
└──────────────────────┴────────────────────────────────────────┘
```

### List All Runs

```bash
python -m app.main --list
```

### Resume an Existing Run

```bash
python -m app.main --resume <run-id>
```

### Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run ClarificationAgent tests only
pytest tests/test_clarification.py -v

# Run with coverage
pytest tests/ --cov=app --cov=agents
```

## Project Structure

```
multiagent-prd/
├── app/
│   ├── __init__.py
│   ├── main.py              # CLI interface ✅
│   ├── config.py            # Configuration management ✅
│   ├── logger.py            # Logging setup ✅
│   ├── state.py             # State schema and persistence ✅
│   └── orchestrator.py      # DAG-based agent coordination ✅
├── agents/
│   ├── __init__.py
│   ├── base_agent.py        # Base agent with ReAct framework ✅
│   ├── clarification.py     # ClarificationAgent ✅
│   ├── planner.py           # PlannerAgent ✅
│   ├── prompts/
│   │   ├── clarification.txt # Clarification prompt ✅
│   │   └── planning.txt      # Planning prompt ✅
│   └── README.md            # Agent documentation ✅
├── tests/
│   ├── __init__.py
│   ├── test_clarification.py # ClarificationAgent tests (11) ✅
│   └── test_planner.py       # PlannerAgent tests (26) ✅
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

## Development Log

### 📅 Day 1 (January 28-29, 2026) ✅ COMPLETE

**What Was Built:**
- ✅ Complete project foundation and scaffolding
- ✅ State management with Pydantic models and JSON persistence
- ✅ Configuration and logging infrastructure
- ✅ BaseAgent class with ReAct framework
- ✅ **ClarificationAgent** - Full implementation with:
  - Structured metadata extraction (domain, tags, users, compliance)
  - OpenAI structured output mode
  - 159-line prompt with 15-domain taxonomy and 5 few-shot examples
  - 11 comprehensive tests (all passing)
- ✅ CLI interface with Rich formatting
- ✅ Verbose mode for detailed agent traces
- ✅ Orchestrator with agent execution loop

**Key Achievements:**
- 349 lines of production-ready agent code
- Full test coverage for ClarificationAgent
- Beautiful table output for extracted metadata
- Fixed infinite loop bug in agent execution
- Complete documentation (agents/README.md, USAGE.md)

**Metrics:**
- Total LOC: ~3,500 lines
- Test Coverage: 11 tests, 100% passing
- API Cost per run: ~$0.01-0.02
- Execution Time: 2-5 seconds (clarification only)

---

### 📅 Day 2 (January 29, 2026) ✅ COMPLETE

**What Was Built:**
- ✅ **PlannerAgent** - Full implementation with:
  - Domain-specific research query generation (15-20 queries per run)
  - 4 query categories: competitor, pain_points, workflow, compliance
  - Priority assignment (high/medium/low)
  - Expected sources tagging (forums, reviews, pricing_pages, etc.)
  - Post-processing for year markers and duplicate detection
  - 437-line prompt with domain-specific competitor lists
  - 26 comprehensive tests (all passing)
- ✅ **DAG Orchestrator** - Complete rewrite with:
  - Task dependency resolution
  - Agent registry with auto-discovery
  - State checkpointing after each agent
  - Retry logic with exponential backoff
- ✅ **Multi-domain testing** across 5 verticals:
  - Fintech (invoicing, expense tracking)
  - Healthcare (telemedicine, patient portals)
  - DevTools (security scanning, CI/CD)
  - Real Estate (CRM, property management)
  - Ecommerce (inventory, order management)

**Key Achievements:**
- 349 lines of PlannerAgent code
- 437-line prompt with 3 few-shot examples
- Query quality: 60-80% include year markers
- Fuzzy duplicate detection (80% threshold)
- All 37 tests passing

**Sample Output:**
```python
state.research_plan.queries = [
  Query(
    id="Q1",
    text="athenahealth vs Kareo pricing small practice 2024",
    category="competitor",
    priority="high",
    expected_sources=["pricing_pages", "comparison_sites"]
  ),
  Query(
    id="Q2",
    text="small clinic EHR implementation problems reddit",
    category="pain_points",
    priority="high",
    expected_sources=["forums"]
  ),
  # ... 13-18 more queries
]
```

---

### 📅 Day 3 (TBD) - ResearcherAgent & Web Search

**Planned:**
- [ ] SearchAgent implementation
- [ ] Web search tool integration (Tavily, Perplexity, or custom)
- [ ] Execute queries from research plan
- [ ] Extract and store evidence with citations
- [ ] Populate `state.evidence[]`

---

### 📅 Day 4 (TBD) - SynthesisAgent

**Planned:**
- [ ] SynthesisAgent implementation
- [ ] Analyze evidence and extract insights
- [ ] Identify pain points, competitors, workflows
- [ ] Populate `state.insights`

---

### 📅 Day 5 (TBD) - PRDWriterAgent

**Planned:**
- [ ] PRDWriterAgent implementation
- [ ] Generate PRD sections with citations
- [ ] Notion markdown formatting
- [ ] Citation management
- [ ] Populate `state.prd`

---

## Development Roadmap

### Phase 1: Foundation ✅ COMPLETE
- [x] State schema and persistence
- [x] Configuration management
- [x] Logging infrastructure
- [x] Base agent with ReAct framework
- [x] CLI interface with Rich output
- [x] Orchestrator with agent execution
- [x] **ClarificationAgent** - Full implementation with tests

### Phase 2: Core Agents (Days 2-3)
- [x] Research Planner Agent ✅
- [x] DAG-based Orchestrator ✅
- [ ] Web Search Tool Integration
- [ ] Search Execution Agent
- [ ] Evidence collection and storage

### Phase 3: PRD Generation (Days 4-5)
- [ ] Insight Synthesis Agent
- [ ] PRD Writer Agent
- [ ] Citation Manager
- [ ] Quality Review Agent
- [ ] Notion Markdown Formatter

### Phase 4: Enhancements (Future)
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
# Run all tests
pytest tests/ -v

# Run ClarificationAgent tests (11 tests, all passing ✅)
pytest tests/test_clarification.py -v

# Run with coverage
pytest tests/ --cov=app --cov=agents --cov-report=html

# Run specific test
pytest tests/test_clarification.py::test_freelance_invoice_tool -v

# Type checking
mypy app/ agents/

# Code formatting
black app/ agents/
```

### Test Coverage (Day 2)

**Total: 37 tests, all passing ✅**

**ClarificationAgent** - 11 tests ✅
- ✅ test_freelance_invoice_tool - Fintech domain extraction
- ✅ test_healthcare_portal - Healthcare domain with compliance
- ✅ test_devtools_security - DevTools domain
- ✅ test_vague_idea - Handles unclear input with questions
- ✅ test_no_clarification_questions - Clear ideas skip questions
- ✅ test_already_clarified - Skips if already run
- ✅ test_api_error_handling - Retry logic with exponential backoff
- ✅ test_invalid_json_response - Handles malformed LLM output
- ✅ test_response_validation - Pydantic validation
- ✅ test_industry_tags_constraints - Min/max validation (2-4 tags)
- ✅ test_clarification_response_model - Model validation

**PlannerAgent** - 26 tests ✅
- ✅ test_fintech_queries - Generates fintech-specific queries
- ✅ test_healthcare_queries - Healthcare domain with HIPAA queries
- ✅ test_devtools_queries - Security/DevOps query generation
- ✅ test_query_count_range - Validates 15-20 query count
- ✅ test_category_distribution - Validates category requirements
- ✅ test_priority_distribution - High/medium/low balance
- ✅ test_duplicate_detection - Fuzzy matching at 80% threshold
- ✅ test_year_markers - 60%+ queries include year
- ✅ test_competitor_names - Named competitors in queries
- ✅ test_expected_sources - Source tagging validation
- ✅ test_skip_if_already_planned - Idempotent execution
- ✅ + 15 additional edge case tests

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
