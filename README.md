# Multi-Agent PRD Generator

A sophisticated tool that generates research-backed Product Requirements Documents (PRDs) using multiple AI agents coordinating through a shared state object. Built with the ReAct (Reasoning + Acting) framework.

## 🎉 Day 3 Complete!

Three agents are live and collecting real research! Here's what's working:

- ✅ **ClarificationAgent** - Extracts structured metadata from product ideas
- ✅ **PlannerAgent** - Generates 15-20 domain-specific research queries
- ✅ **ResearcherAgent** - Executes queries and collects 50-80 evidence sources
- ✅ **Web Search** - Tavily API integration with caching
- ✅ **Content Extraction** - Jina Reader for clean markdown content
- ✅ **Credibility Scoring** - Domain, recency, and content quality signals
- ✅ **Evidence Deduplication** - MD5 + SimHash hybrid approach
- ✅ **DAG Orchestrator** - Manages task dependencies and agent sequencing
- ✅ **Test Suite** - 185+ passing tests
- ⏳ **SynthesisAgent** - Coming in Day 4
- ⏳ **PRDWriterAgent** - Coming in Day 5

## Overview

This tool transforms a simple product idea into a comprehensive PRD by:
- ✅ **Clarifying ambiguous requirements** through intelligent metadata extraction
- ✅ **Planning targeted research** with domain-specific queries and competitor analysis
- ✅ **Conducting web research** to gather 50-80 evidence sources per run
- ✅ **Scoring source credibility** based on domain, recency, and content quality
- ⏳ Analyzing competitors, pain points, and user workflows (Day 4)
- ⏳ Synthesizing findings into a well-structured PRD with citations (Day 5)

## Features

- **Multi-Agent Architecture**: Specialized agents work together to handle different aspects of PRD generation
- **ReAct Framework**: Each agent uses a Think-Act-Observe-Update-Reflect loop for intelligent decision-making
- **Research-Backed**: All claims in the PRD are backed by web research with proper citations
- **Stateful & Resumable**: Complete execution state is persisted, allowing runs to be paused and resumed
- **Production-Ready**: Comprehensive error handling, logging, retry logic, and type safety
- **Rich CLI**: Beautiful command-line interface with progress tracking and formatted output

### ✅ Research Execution (New in Day 3)

- **Web Search**: Tavily API integration (1,000 free searches/month)
- **Content Extraction**: Jina Reader for clean markdown from any URL
- **50-80 sources** collected per product idea
- **Source Credibility Scoring**:
  - Domain reputation (high: .gov, .edu, official docs)
  - Recency (newer content weighted higher)
  - Content quality signals (statistics, research, depth)
- **Evidence Typing**: article, forum, review, pricing, docs
- **Automatic Deduplication**: MD5 + SimHash for exact and near-duplicate detection
- **Smart Caching**: 24hr TTL to minimize API calls

## Current Architecture (Day 3)

```
User Input: "Build a HIPAA-compliant patient portal"
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    DAG Orchestrator                         │
│  - Task dependency resolution                               │
│  - Agent scheduling & retry logic                           │
│  - State checkpointing after each agent                     │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
│ Clarification    │ │   Planner        │ │   Researcher     │
│ Agent ✅         │ │   Agent ✅       │ │   Agent ✅       │
│                  │ │                  │ │                  │
│ Extracts:        │ │ Generates:       │ │ Executes:        │
│ - domain         │→│ - 15-20 queries  │→│ - Web search     │
│ - industry_tags  │ │ - 4 categories   │ │ - Content fetch  │
│ - target_user    │ │ - priorities     │ │ - Credibility    │
│ - compliance     │ │ - sources        │ │ - Deduplication  │
└──────────────────┘ └──────────────────┘ └──────────────────┘
                                                   │
                          ┌────────────────────────┘
                          ▼
                 ┌──────────────────┐
                 │   Shared State   │
                 │                  │
                 │  ✅ metadata     │
                 │  ✅ research_plan│
                 │  ✅ evidence     │  ← 50-80 sources
                 │  ⏳ insights     │
                 │  ⏳ prd          │
                 └──────────────────┘
                          │
                          ▼
             Saved to: data/runs/{run_id}.json
```

## Evidence Quality

Our research collects high-quality, diverse sources:

**Source Distribution (typical run):**
- 📄 Articles: 35-45%
- 💬 Forums (Reddit, HN, Stack Overflow): 20-30%
- ⭐ Reviews (G2, Capterra, TrustRadius): 15-20%
- 💰 Pricing pages: 10-15%
- 📚 Documentation: 5-10%

**Credibility Distribution:**
- 🟢 High credibility: 20-30% (.gov, .edu, industry reports, official docs)
- 🟡 Medium credibility: 50-60% (tech news, business sites, review platforms)
- 🔴 Low credibility: 15-25% (forums, social media - still valuable for pain points!)

**Deduplication:**
- URL canonicalization (removes tracking params, www variants)
- MD5 hash for exact content matches
- SimHash for near-duplicate detection (paraphrased content)
- Fuzzy title matching (85% similarity threshold)

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

### Inspect Evidence (New in Day 3)

```bash
# View all evidence from a run
python -m app.main --inspect <run-id>

# Filter by evidence type
python -m app.main --inspect <run-id> --type forum
python -m app.main --inspect <run-id> --type review

# Filter by credibility
python -m app.main --inspect <run-id> --credibility high

# View specific evidence details
python -m app.main --inspect <run-id> --evidence-id E5

# Combine filters
python -m app.main --inspect <run-id> --type docs --credibility high
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
│   ├── researcher.py        # ResearcherAgent ✅ (NEW)
│   ├── prompts/
│   │   ├── clarification.txt # Clarification prompt ✅
│   │   ├── planning.txt      # Planning prompt ✅
│   │   └── researcher.txt    # Research prompt (placeholder) ✅
│   └── README.md            # Agent documentation ✅
├── tools/                   # Research tools (NEW)
│   ├── __init__.py          # Package exports ✅
│   ├── web_search.py        # Tavily API integration ✅
│   ├── fetch_url.py         # Jina Reader content extraction ✅
│   ├── credibility.py       # Source credibility scoring ✅
│   └── dedupe.py            # Evidence deduplication ✅
├── tests/
│   ├── __init__.py
│   ├── test_clarification.py # ClarificationAgent tests (11) ✅
│   ├── test_planner.py       # PlannerAgent tests (26) ✅
│   ├── test_researcher.py    # ResearcherAgent tests (35) ✅
│   ├── test_web_search.py    # Web search tests (28) ✅
│   ├── test_fetch_url.py     # Content fetch tests (38) ✅
│   ├── test_credibility.py   # Credibility tests (38) ✅
│   └── test_dedupe.py        # Deduplication tests (46) ✅
├── data/
│   ├── runs/                # Saved run states (auto-created)
│   ├── cache/               # API response cache (auto-created)
│   │   ├── search/          # Search results cache
│   │   └── content/         # Fetched content cache
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

### 📅 Day 3 (February 2, 2026) ✅ COMPLETE

**What Was Built:**
- ✅ **ResearcherAgent** - Full implementation with:
  - Executes all queries from research plan
  - Collects 50-80 evidence sources per run
  - Parallel URL fetching (3 concurrent)
  - Rich progress bar with real-time status
  - Evidence type inference from URLs
  - 35 comprehensive tests (all passing)
- ✅ **Web Search Tool** (Tavily API):
  - Advanced search with domain filtering
  - Rate limiting (0.5s between requests)
  - Exponential backoff retry (3 attempts)
  - File-based caching (24hr TTL)
  - 28 tests (all passing)
- ✅ **Content Fetcher** (Jina Reader):
  - Clean markdown extraction from any URL
  - Smart truncation at sentence boundaries
  - Metadata extraction (title, author, date)
  - Caching (48hr TTL)
  - 38 tests (all passing)
- ✅ **Credibility Scorer**:
  - Domain reputation tiers (50+ high, 30+ medium, 15+ low)
  - Recency scoring (favors recent content)
  - Content quality signals (research, statistics, depth)
  - Spam/clickbait detection
  - 38 tests (all passing)
- ✅ **Evidence Deduplicator**:
  - URL canonicalization (tracking params, www, fragments)
  - MD5 hash for exact content matches
  - SimHash for near-duplicate detection
  - Fuzzy title matching (85% threshold)
  - 46 tests (all passing)
- ✅ **CLI Enhancements**:
  - `--inspect <run_id>` to view evidence
  - Filter by `--type` (article, forum, docs, etc.)
  - Filter by `--credibility` (high, medium, low)
  - View details with `--evidence-id E5`

**Key Achievements:**
- Full research pipeline: Idea → Metadata → Queries → Evidence
- 185+ tests all passing
- 4 production-ready research tools
- Smart caching saves API calls
- Evidence ready for Day 4 analysis

**Metrics:**
- Total LOC: ~6,000+ lines
- Test Coverage: 185+ tests, 100% passing
- API Cost per run: ~$0.10-0.20 (search + content)
- Execution Time: 2-5 minutes (full research)

**Sample Evidence Output:**
```
Evidence Collected: 67 sources

By Type: {'article': 28, 'forum': 18, 'review': 12, 'pricing': 6, 'docs': 3}
By Credibility: {'high': 15, 'medium': 41, 'low': 11}

┌─────┬─────────┬──────────┬─────────────────────────────────────────┐
│ ID  │ Type    │ Cred     │ Title                                   │
├─────┼─────────┼──────────┼─────────────────────────────────────────┤
│ E1  │ article │ high     │ Best Patient Scheduling Software 2024   │
│ E2  │ review  │ medium   │ athenahealth vs Kareo - G2 Comparison   │
│ E3  │ forum   │ low      │ HIPAA compliant messaging? : r/healthIT │
│ ... │ ...     │ ...      │ ...                                     │
└─────┴─────────┴──────────┴─────────────────────────────────────────┘
```

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

### Phase 2: Core Agents (Days 2-3) ✅ COMPLETE
- [x] Research Planner Agent ✅
- [x] DAG-based Orchestrator ✅
- [x] Web Search Tool (Tavily API) ✅
- [x] Content Extraction (Jina Reader) ✅
- [x] Credibility Scoring ✅
- [x] Evidence Deduplication ✅
- [x] **ResearcherAgent** - Full implementation with tests ✅

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
| `TAVILY_API_KEY` | Tavily search API key (required for research) | - |
| `JINA_API_KEY` | Jina Reader API key (optional, higher rate limits) | - |
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

### Test Coverage (Day 3)

**Total: 185+ tests, all passing ✅**

**ClarificationAgent** - 11 tests ✅
- ✅ Metadata extraction across 5 domains
- ✅ Compliance detection (HIPAA, GDPR, SOC2)
- ✅ Error handling and retry logic

**PlannerAgent** - 26 tests ✅
- ✅ Query generation (15-20 per run)
- ✅ Category distribution validation
- ✅ Duplicate detection and year markers

**ResearcherAgent** - 35 tests ✅
- ✅ Query execution and evidence collection
- ✅ Type inference (article, forum, docs, pricing, review)
- ✅ State updates and task management
- ✅ Error handling for failed searches/fetches

**Web Search Tool** - 28 tests ✅
- ✅ Tavily API integration
- ✅ Caching and rate limiting
- ✅ Retry with exponential backoff

**Content Fetcher** - 38 tests ✅
- ✅ Jina Reader content extraction
- ✅ Smart truncation and metadata parsing
- ✅ Error handling and caching

**Credibility Scorer** - 38 tests ✅
- ✅ Domain tier classification
- ✅ Recency and content quality scoring
- ✅ Spam/clickbait detection

**Evidence Deduplicator** - 46 tests ✅
- ✅ URL canonicalization
- ✅ MD5 hash matching
- ✅ SimHash near-duplicate detection
- ✅ Fuzzy title matching

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
