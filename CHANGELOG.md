# Changelog

## 2026-01-29 - Enhanced CLI Output & Verbose Mode

### Added
- **Metadata Display Table**: Extracted metadata now shown in a beautiful Rich table
  - Domain, Industry Tags, Target User, Geography, Compliance
  - Automatically displayed after successful clarification
  
- **Verbose Mode (`--verbose` / `-v`)**: Show detailed agent execution trace
  - Displays complete agent trace in a formatted table
  - Shows all agent actions with turn numbers
  - Usage: `python -m app.main "idea" --verbose`

- **Next Steps Indicator**: Shows what agent will run next
  - Currently displays: "Next Step: Run ResearchPlannerAgent (coming soon)"
  
- **Demo Script (`demo.sh`)**: Automated demonstration of all features

### Enhanced
- `display_run_summary()` function now shows:
  - Extracted metadata in formatted table (if available)
  - Detailed agent trace (when --verbose flag used)
  - Clearer status indicators with emoji
  
- Help message updated with verbose flag example

### Fixed
- **Infinite Loop Bug**: ClarificationAgent was running infinitely
  - Root cause: Agent set status to "pending" when it had questions
  - Orchestrator saw status != "confirmed" and kept re-running
  - Solution: Check if `domain` is populated (indicates agent already ran)
  - Updated both `agents/clarification.py` and `app/orchestrator.py`
  
- Test suite updated to work with new logic
  - `test_already_clarified` now sets domain when marking as clarified

### Output Examples

**Standard Output:**
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

**Verbose Output:**
```
Agent Trace:
┏━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Turn   ┃ Agent           ┃ Action                                  ┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 1      │ clarification   │ Loading clarification prompt            │
│ 2      │ clarification   │ Calling LLM with structured output mode │
│ 3      │ clarification   │ Parsing LLM response                    │
│ 4      │ clarification   │ Updating state metadata                 │
│ 5      │ clarification   │ Made 3 assumptions                      │
│ 6      │ clarification   │ Generated 3 clarification questions     │
│ 7      │ clarification   │ Clarification completed successfully    │
└────────┴─────────────────┴─────────────────────────────────────────┘
```

### Files Modified
- `app/main.py`:
  - Enhanced `display_run_summary()` with metadata table and verbose trace
  - Added `--verbose` flag to argument parser
  - Updated help examples
  - Added "Next Steps" indicator

- `agents/clarification.py`:
  - Fixed skip logic to check if domain is populated

- `app/orchestrator.py`:
  - Fixed agent selection to check domain instead of status

- `tests/test_clarification.py`:
  - Updated `test_already_clarified` to set domain

### Testing
All 11 tests passing ✅

### Usage

```bash
# Standard run
python -m app.main "Build a project management tool"

# Verbose output with agent trace
python -m app.main "Build a project management tool" --verbose

# Short form
python -m app.main "Build a project management tool" -v

# List all runs
python -m app.main --list

# Resume a run
python -m app.main --resume <run-id>

# Run demo
./demo.sh
```
