# Usage Guide - Multi-Agent PRD Generator

## Quick Start

### 1. Installation

```bash
# Clone and navigate to project
cd multiagent-prd

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-...
```

### 3. Run Your First PRD Generation

```bash
# Generate a PRD
python -m app.main "Build a project management tool for remote teams"
```

## Current Capabilities (Phase 1)

The system currently includes:

✅ **ClarificationAgent** - Extracts structured metadata from product ideas

The agent will analyze your idea and extract:
- **Domain**: Primary product category (fintech, healthcare, devtools, etc.)
- **Industry Tags**: 2-4 specific tags for research targeting
- **Target User**: Specific audience description
- **Geography**: Geographic focus or market
- **Compliance Contexts**: Relevant regulations (HIPAA, GDPR, SOC2, etc.)

### Example Output

```bash
$ python -m app.main "Build a HIPAA-compliant telemedicine platform"

============================================================
              Starting New PRD Generation
============================================================

Product Idea: Build a HIPAA-compliant telemedicine platform
✓ Created new run: abc123-def456-789...
ℹ Registering agents...
ℹ Registered agent: clarification
ℹ Starting orchestration...

============================================================
                      Run Summary
============================================================

Run ID: abc123-def456-789...
Status: done
Created: 2026-01-28 10:30:15

Product Idea:
Build a HIPAA-compliant telemedicine platform

Metadata:
  Domain: healthcare
  Industry Tags: telemedicine, telehealth, patient_engagement
  Target User: healthcare providers and patients
  Geography: US
  Compliance: HIPAA, state_medical_boards

Tasks: 1 total
  - Pending: 0
  - In Progress: 0
  - Done: 1
  - Blocked: 0

Agent Actions: 7
  Last: clarification - Clarification completed successfully

✓ PRD generation completed successfully!
ℹ Output saved to: data/runs/abc123-def456-789....json
```

## CLI Commands

### Start New Run

```bash
python -m app.main "your product idea here"
```

**Examples**:

```bash
# Fintech product
python -m app.main "Invoice tracking tool for freelance designers"

# Healthcare product
python -m app.main "Patient portal for small medical clinics"

# DevTools product
python -m app.main "Security vulnerability scanner for CI/CD pipelines"

# Productivity tool
python -m app.main "AI scheduling assistant for executives"
```

### Resume Existing Run

```bash
python -m app.main --resume <run-id>
```

### List All Runs

```bash
python -m app.main --list
```

## Running the Example Script

See the ClarificationAgent in action with example product ideas:

```bash
python example.py
```

This will demonstrate metadata extraction for:
1. Freelance invoice tracking tool
2. HIPAA-compliant patient portal
3. Security vulnerability platform
4. AI scheduling assistant

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test File

```bash
pytest tests/test_clarification.py -v
```

### Run with Coverage

```bash
pytest tests/ --cov=app --cov=agents --cov-report=html
```

### Run Specific Test

```bash
pytest tests/test_clarification.py::test_freelance_invoice_tool -v
```

## Understanding the Output

### State File

Each run creates a JSON file in `data/runs/`:

```bash
data/runs/abc123-def456-789....json
```

**Structure**:
```json
{
  "run_id": "abc123...",
  "created_at": "2026-01-28T10:30:15",
  "status": "done",
  "metadata": {
    "raw_idea": "Build a...",
    "domain": "healthcare",
    "industry_tags": ["telemedicine", ...],
    "target_user": "healthcare providers",
    "geography": "US",
    "compliance_contexts": ["HIPAA"]
  },
  "task_board": [...],
  "agent_trace": [...]
}
```

### Logs

Application logs are saved to `data/logs/app.log`:

```bash
tail -f data/logs/app.log
```

## Advanced Usage

### Custom Configuration

Edit `.env` to customize:

```bash
# Use different model
OPENAI_MODEL=gpt-4o-mini

# Adjust logging
LOG_LEVEL=DEBUG

# Change retry behavior
MAX_RETRIES=5
RETRY_DELAY=2

# Custom directories
OUTPUT_DIR=my_runs
LOG_DIR=my_logs
```

### Programmatic Usage

```python
from openai import OpenAI
from agents.clarification import ClarificationAgent
from app.config import get_config
from app.state import create_new_state, save_state

# Setup
config = get_config()
client = OpenAI(api_key=config.openai_api_key)

# Create state
state = create_new_state("Your product idea")

# Run clarification
agent = ClarificationAgent("clarification", client)
state = agent.run(state)

# Save results
save_state(state)

# Access metadata
print(f"Domain: {state.metadata.domain}")
print(f"Target User: {state.metadata.target_user}")
```

### Inspecting State

```python
from app.state import load_state

# Load a run
state = load_state("abc123-def456-789...")

# Check metadata
print(state.metadata.domain)
print(state.metadata.industry_tags)

# View agent actions
for entry in state.agent_trace:
    print(f"[{entry.agent}] {entry.action}")
    if entry.details:
        print(f"  Details: {entry.details}")

# Check tasks
for task in state.task_board:
    print(f"[{task.status}] {task.description}")
```

## Troubleshooting

### Missing API Key

```
Error: Required environment variable OPENAI_API_KEY is not set
```

**Solution**: Add `OPENAI_API_KEY=sk-...` to your `.env` file

### Import Errors

```
ModuleNotFoundError: No module named 'openai'
```

**Solution**: Activate virtual environment and install dependencies:
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### API Rate Limits

```
OpenAI API error: Rate limit exceeded
```

**Solution**: The agent will automatically retry with exponential backoff. If it persists, wait a minute or upgrade your OpenAI plan.

### Validation Errors

```
Failed to validate response: ...
```

**Solution**: This usually means the LLM returned unexpected output. Check logs in `data/logs/app.log` for details. The agent will retry automatically.

## What's Next?

Currently, only the ClarificationAgent is implemented. The system will expand to include:

**Phase 2** (Coming Next):
- ResearchPlannerAgent - Generates targeted research queries
- SearchAgent - Executes web searches
- SynthesisAgent - Analyzes findings

**Phase 3**:
- PRDWriterAgent - Generates PRD sections
- CitationAgent - Manages evidence citations
- ReviewAgent - Quality checks

See [README.md](README.md) for the full roadmap.

## Getting Help

- **Documentation**: Read `README.md` and `agents/README.md`
- **Logs**: Check `data/logs/app.log` for detailed execution logs
- **State Files**: Inspect `data/runs/*.json` to see exact state
- **Tests**: Look at `tests/` for usage examples

## Cost Estimation

**Per Run** (with ClarificationAgent only):
- API Calls: 1-2 calls
- Tokens: ~500-1000 tokens
- Cost: ~$0.01-0.02 per run

**Note**: Costs will increase as more agents are added to the pipeline.
