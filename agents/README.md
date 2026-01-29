# Agents Documentation

This directory contains all the agents that make up the multi-agent PRD generator system.

## Agent Architecture

All agents inherit from `BaseAgent` and implement the **ReAct framework** (Reasoning + Acting):

1. **Think**: Analyze the current state
2. **Act**: Call LLM with appropriate prompts
3. **Observe**: Parse and validate the response
4. **Update**: Modify the shared state
5. **Reflect**: Log actions to the agent trace

## Available Agents

### 1. ClarificationAgent

**Purpose**: Extracts structured metadata from the user's raw product idea.

**Input**:
- `state.metadata.raw_idea` (string)

**Output**: Updates `state.metadata` with:
- `domain` - Primary product domain (e.g., "fintech", "healthcare")
- `industry_tags` - 2-4 specific industry tags
- `target_user` - Specific target audience description
- `geography` - Geographic focus ("US", "EU", "global", etc.)
- `compliance_contexts` - Relevant regulations (HIPAA, GDPR, SOC2, etc.)

**Features**:
- Uses OpenAI structured output mode for reliable parsing
- Makes explicit assumptions when details are unclear
- Generates clarification questions for ambiguous requirements
- Comprehensive error handling with retries
- Detailed logging and tracing

**Example**:

```python
from openai import OpenAI
from agents.clarification import ClarificationAgent
from app.state import create_new_state

# Create state with product idea
state = create_new_state("Build a tool for freelance designers to track invoices")

# Initialize agent
client = OpenAI(api_key="your-key")
agent = ClarificationAgent("clarification", client)

# Run agent
state = agent.run(state)

# Access extracted metadata
print(state.metadata.domain)  # "fintech"
print(state.metadata.target_user)  # "freelance designers and creative professionals"
```

**Domain Taxonomy**:
The agent uses a standardized domain taxonomy including:
- `fintech` - Payments, invoicing, banking, trading
- `healthcare` - EMR, telemedicine, health tracking
- `edtech` - Learning platforms, course management
- `devtools` - CI/CD, monitoring, debugging, security
- `productivity` - Project management, automation
- `ecommerce` - Online stores, inventory, fulfillment
- And 10+ more categories

**Testing**:
```bash
# Run ClarificationAgent tests
pytest tests/test_clarification.py -v

# Run with coverage
pytest tests/test_clarification.py --cov=agents.clarification
```

**Prompt Template**: `agents/prompts/clarification.txt`

The prompt includes:
- Role definition
- Task instructions
- Domain taxonomy
- 5 few-shot examples
- Specific rules and guidelines

## Creating a New Agent

1. **Create agent file** in `agents/`:

```python
from agents.base_agent import BaseAgent
from app.state import State

class MyAgent(BaseAgent):
    def run(self, state: State) -> State:
        # 1. Think
        if self._should_skip(state):
            return state

        # 2. Act
        prompt = self._load_prompt()
        response = self._call_llm([{"role": "user", "content": prompt}])

        # 3. Observe
        observations = self._observe(response)

        # 4. Update
        state = self._update_state(state, observations)

        # 5. Reflect
        self._log_action(state, "Completed task")

        return state
```

2. **Create prompt template** in `agents/prompts/my_agent.txt`

3. **Register agent** in `app/main.py`:

```python
from agents.my_agent import MyAgent

orchestrator.register_agent(MyAgent("my_agent", client))
```

4. **Write tests** in `tests/test_my_agent.py`

## Best Practices

### Error Handling
- Always use try/except blocks
- Mark tasks as "blocked" on failure
- Log errors with context
- Use retry logic for API calls (inherited from BaseAgent)

### State Management
- Never mutate state in-place without returning it
- Always save state after major operations
- Use Pydantic models for validation
- Keep state updates atomic

### Logging
- Log at appropriate levels (DEBUG, INFO, WARNING, ERROR)
- Use `_log_action()` to add to agent trace
- Include relevant details in trace entries
- Log both successes and failures

### Testing
- Mock OpenAI API calls
- Test happy path and error cases
- Verify state updates
- Check task board management
- Test validation logic

### Prompts
- Use clear role definitions
- Include few-shot examples (3-5)
- Specify output format explicitly
- Add rules and constraints
- Use placeholders (e.g., `{{raw_idea}}`) for dynamic content

## Agent Communication

Agents communicate through the **shared state object**. Each agent:

1. Reads relevant parts of the state
2. Makes decisions based on state
3. Updates specific state fields
4. Adds tasks to the task board
5. Logs actions to agent trace

**Example state flow**:
```
ClarificationAgent → state.metadata (populated)
                  → state.task_board (task added)
                  → state.agent_trace (actions logged)

ResearchPlannerAgent ← reads state.metadata
                     → state.research_plan (queries added)
                     → state.task_board (task added)

SearchAgent ← reads state.research_plan
            → state.evidence (research results added)
```

## Debugging Agents

### View Agent Trace
```python
from app.state import load_state

state = load_state("run-id")
for entry in state.agent_trace:
    print(f"[{entry.agent}] Turn {entry.turn}: {entry.action}")
    if entry.details:
        print(f"  Details: {entry.details}")
```

### Check Task Board
```python
for task in state.task_board:
    print(f"[{task.status}] {task.description} (owner: {task.owner})")
```

### Enable Debug Logging
```bash
# In .env file
LOG_LEVEL=DEBUG
```

## Performance Considerations

- **API Costs**: Each agent may make 1-3 API calls (~$0.01-0.05 per run)
- **Latency**: Typical agent execution: 2-5 seconds
- **Retries**: Max 3 retries with exponential backoff
- **Token Usage**: Logged in agent trace for monitoring

## Future Agents (Roadmap)

- [ ] **ResearchPlannerAgent** - Generates research queries
- [ ] **SearchAgent** - Executes web searches
- [ ] **SynthesisAgent** - Analyzes research findings
- [ ] **PRDWriterAgent** - Generates PRD sections
- [ ] **CitationAgent** - Manages evidence citations
- [ ] **ReviewAgent** - Quality checks the PRD
