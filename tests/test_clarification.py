"""Tests for the ClarificationAgent.

These tests verify that the clarification agent correctly extracts
structured metadata from various types of product ideas.
"""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from openai import OpenAI

from agents.clarification import ClarificationAgent, ClarificationResponse
from app.state import Metadata, State, create_new_state


class MockOpenAIResponse:
    """Mock response from OpenAI API."""

    def __init__(self, content: str):
        self.choices = [
            MagicMock(
                message=MagicMock(content=content, role="assistant"),
                finish_reason="stop"
            )
        ]
        self.usage = MagicMock(
            prompt_tokens=100,
            completion_tokens=200,
            total_tokens=300
        )


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    return MagicMock(spec=OpenAI)


@pytest.fixture
def clarification_agent(mock_openai_client):
    """Create a ClarificationAgent instance with mocked client."""
    return ClarificationAgent("clarification", mock_openai_client)


def create_mock_response(data: dict) -> MockOpenAIResponse:
    """Create a mock OpenAI response with the given data.

    Args:
        data: Dictionary to serialize as JSON response

    Returns:
        MockOpenAIResponse instance
    """
    return MockOpenAIResponse(json.dumps(data))


def test_freelance_invoice_tool(clarification_agent, mock_openai_client):
    """Test clarification of a freelance invoice tracking tool."""
    # Setup
    idea = "Build a tool for freelance designers to track invoices and expenses"
    state = create_new_state(idea)

    mock_data = {
        "domain": "fintech",
        "industry_tags": ["invoicing", "freelance_tools", "expense_tracking"],
        "target_user": "freelance designers and creative professionals",
        "geography": "global",
        "compliance_contexts": ["tax_reporting"],
        "assumptions": [
            "Assuming individual freelancers, not agencies",
            "Assuming need for tax/1099 support"
        ],
        "clarification_questions": [
            "Do you need multi-currency support?",
            "Should it integrate with accounting software (QuickBooks, Xero)?"
        ]
    }

    mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_data)

    # Execute
    result_state = clarification_agent.run(state)

    # Assert
    assert result_state.metadata.domain == "fintech"
    assert "invoicing" in result_state.metadata.industry_tags
    assert "freelance_tools" in result_state.metadata.industry_tags
    assert result_state.metadata.target_user == "freelance designers and creative professionals"
    assert result_state.metadata.geography == "global"
    assert "tax_reporting" in result_state.metadata.compliance_contexts
    assert result_state.metadata.clarification_status == "pending"  # Has questions

    # Verify task was created and marked done
    assert len(result_state.task_board) > 0
    task = result_state.task_board[0]
    assert task.owner == "clarification"
    assert task.status == "done"

    # Verify agent trace
    assert len(result_state.agent_trace) > 0
    assert any("completed successfully" in entry.action.lower() for entry in result_state.agent_trace)


def test_healthcare_portal(clarification_agent, mock_openai_client):
    """Test clarification of a HIPAA-compliant patient portal."""
    # Setup
    idea = "HIPAA-compliant patient portal for small clinics"
    state = create_new_state(idea)

    mock_data = {
        "domain": "healthcare",
        "industry_tags": ["patient_engagement", "EMR", "telehealth"],
        "target_user": "small medical clinics (2-10 providers)",
        "geography": "US",
        "compliance_contexts": ["HIPAA", "state_medical_boards"],
        "assumptions": [
            "Assuming EHR integration needed",
            "Assuming appointment scheduling is core feature"
        ],
        "clarification_questions": [
            "Which EHR systems should integrate with?",
            "Do you need telehealth video calls?"
        ]
    }

    mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_data)

    # Execute
    result_state = clarification_agent.run(state)

    # Assert
    assert result_state.metadata.domain == "healthcare"
    assert "patient_engagement" in result_state.metadata.industry_tags
    assert "EMR" in result_state.metadata.industry_tags
    assert result_state.metadata.target_user == "small medical clinics (2-10 providers)"
    assert result_state.metadata.geography == "US"
    assert "HIPAA" in result_state.metadata.compliance_contexts
    assert "state_medical_boards" in result_state.metadata.compliance_contexts


def test_devtools_security(clarification_agent, mock_openai_client):
    """Test clarification of a security vulnerability platform."""
    # Setup
    idea = "Platform to help engineers find and fix security vulnerabilities"
    state = create_new_state(idea)

    mock_data = {
        "domain": "devtools",
        "industry_tags": ["security", "code_analysis", "vulnerability_management"],
        "target_user": "software engineers and security teams",
        "geography": "global",
        "compliance_contexts": ["SOC2", "data_privacy"],
        "assumptions": [
            "Assuming integration with CI/CD pipelines",
            "Assuming need for SAST/DAST scanning"
        ],
        "clarification_questions": [
            "What languages/frameworks to support?",
            "Should it include dependency vulnerability scanning?"
        ]
    }

    mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_data)

    # Execute
    result_state = clarification_agent.run(state)

    # Assert
    assert result_state.metadata.domain == "devtools"
    assert "security" in result_state.metadata.industry_tags
    assert "vulnerability_management" in result_state.metadata.industry_tags
    assert result_state.metadata.target_user == "software engineers and security teams"
    assert "SOC2" in result_state.metadata.compliance_contexts


def test_vague_idea(clarification_agent, mock_openai_client):
    """Test clarification of a vague product idea."""
    # Setup
    idea = "Make collaboration better for teams"
    state = create_new_state(idea)

    mock_data = {
        "domain": "productivity",
        "industry_tags": ["collaboration", "team_communication"],
        "target_user": "remote teams and distributed organizations",
        "geography": "global",
        "compliance_contexts": ["data_privacy"],
        "assumptions": [
            "Assuming async communication is primary use case",
            "Assuming integration with existing tools is needed",
            "Assuming teams are 5-50 people"
        ],
        "clarification_questions": [
            "What type of collaboration: messaging, project management, or document editing?",
            "What is the primary pain point you're solving?",
            "Who are the competitors you want to differentiate from?"
        ]
    }

    mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_data)

    # Execute
    result_state = clarification_agent.run(state)

    # Assert - Should still extract reasonable metadata
    assert result_state.metadata.domain == "productivity"
    assert len(result_state.metadata.industry_tags) >= 2
    assert result_state.metadata.target_user != ""
    assert len(result_state.metadata.industry_tags) <= 4  # Max 4 tags

    # Should have clarification questions for vague idea
    assert result_state.metadata.clarification_status == "pending"


def test_no_clarification_questions(clarification_agent, mock_openai_client):
    """Test clarification when the idea is very clear."""
    # Setup
    idea = "HIPAA-compliant telemedicine platform for dermatology practices in California with Epic EHR integration"
    state = create_new_state(idea)

    mock_data = {
        "domain": "healthcare",
        "industry_tags": ["telemedicine", "dermatology", "EMR_integration"],
        "target_user": "dermatology practices in California",
        "geography": "US",
        "compliance_contexts": ["HIPAA", "California_medical_board"],
        "assumptions": [
            "Assuming video consultation is primary feature",
            "Assuming need for image capture and storage"
        ],
        "clarification_questions": []  # No questions needed
    }

    mock_openai_client.chat.completions.create.return_value = create_mock_response(mock_data)

    # Execute
    result_state = clarification_agent.run(state)

    # Assert
    assert result_state.metadata.clarification_status == "confirmed"  # No questions = confirmed


def test_already_clarified(clarification_agent, mock_openai_client):
    """Test that agent skips if clarification already confirmed."""
    # Setup
    state = create_new_state("Some idea")
    state.metadata.clarification_status = "confirmed"
    state.metadata.domain = "fintech"  # Set domain to indicate agent already ran

    # Execute
    result_state = clarification_agent.run(state)

    # Assert - Should not call LLM
    mock_openai_client.chat.completions.create.assert_not_called()

    # Should have trace entry indicating skip
    assert any("skipped" in entry.action.lower() for entry in result_state.agent_trace)


def test_api_error_handling(clarification_agent, mock_openai_client):
    """Test handling of API errors with retries."""
    # Setup
    state = create_new_state("Test idea")

    # Mock API to raise error
    from openai import OpenAIError
    mock_openai_client.chat.completions.create.side_effect = OpenAIError("API Error")

    # Execute & Assert
    with pytest.raises(OpenAIError):
        clarification_agent.run(state)

    # Should have attempted retries (default is 3)
    assert mock_openai_client.chat.completions.create.call_count == 3

    # Task should be marked as blocked
    assert len(state.task_board) > 0
    assert state.task_board[0].status == "blocked"


def test_invalid_json_response(clarification_agent, mock_openai_client):
    """Test handling of invalid JSON response."""
    # Setup
    state = create_new_state("Test idea")

    # Mock invalid JSON response
    mock_openai_client.chat.completions.create.return_value = MockOpenAIResponse(
        "This is not valid JSON"
    )

    # Execute & Assert
    with pytest.raises(Exception):
        clarification_agent.run(state)


def test_response_validation(clarification_agent, mock_openai_client):
    """Test Pydantic validation of response."""
    # Setup
    state = create_new_state("Test idea")

    # Mock response with invalid data (missing required fields)
    invalid_data = {
        "domain": "fintech",
        # Missing required fields
    }

    mock_openai_client.chat.completions.create.return_value = create_mock_response(invalid_data)

    # Execute & Assert
    with pytest.raises(Exception):
        clarification_agent.run(state)


def test_industry_tags_constraints(clarification_agent, mock_openai_client):
    """Test that industry_tags respects min/max constraints."""
    # Setup
    state = create_new_state("Test idea")

    # Test with too many tags (more than 4)
    invalid_data = {
        "domain": "fintech",
        "industry_tags": ["tag1", "tag2", "tag3", "tag4", "tag5"],  # 5 tags
        "target_user": "test users",
        "geography": "global",
        "compliance_contexts": [],
        "assumptions": [],
        "clarification_questions": []
    }

    mock_openai_client.chat.completions.create.return_value = create_mock_response(invalid_data)

    # Execute & Assert - Should fail validation
    with pytest.raises(Exception):
        clarification_agent.run(state)


def test_clarification_response_model():
    """Test the ClarificationResponse Pydantic model directly."""
    # Valid data
    valid_data = {
        "domain": "fintech",
        "industry_tags": ["invoicing", "payments"],
        "target_user": "freelancers",
        "geography": "US",
        "compliance_contexts": ["tax_reporting"],
        "assumptions": ["Some assumption"],
        "clarification_questions": ["Question?"]
    }

    response = ClarificationResponse(**valid_data)
    assert response.domain == "fintech"
    assert len(response.industry_tags) == 2

    # Test with minimal tags
    with pytest.raises(Exception):
        ClarificationResponse(
            domain="fintech",
            industry_tags=["only_one"],  # Should fail, min is 2
            target_user="users",
            geography="global"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
