"""Unit tests for Arcane agents and crews."""

import os
import pytest
from unittest.mock import patch

# CrewAI requires an LLM provider string or BaseLLM. We set a fake
# OPENAI_API_KEY env var so CrewAI can initialise agents during tests
# without actually calling any API.
os.environ.setdefault("OPENAI_API_KEY", "sk-test-fake-key-for-unit-tests")


class TestAgentCreation:
    """Tests for agent factory functions."""

    def test_create_planner_agent(self):
        from arcane.agents.planner import create_planner_agent

        agent = create_planner_agent()
        assert agent.role == "Senior Research Strategist"
        assert agent.allow_delegation is False

    def test_create_critic_agent(self):
        from arcane.agents.critic import create_critic_agent

        agent = create_critic_agent()
        assert agent.role == "Research Quality Auditor"
        assert "evaluate" in agent.backstory.lower() or "rubric" in agent.backstory.lower()

    def test_create_synthesizer_agent(self):
        from arcane.agents.synthesizer import create_synthesizer_agent

        agent = create_synthesizer_agent()
        assert agent.role == "Research Synthesis Expert"

    def test_create_query_generator_agent(self):
        from arcane.agents.query_generator import create_query_generator_agent

        agent = create_query_generator_agent()
        assert agent.role == "Search Query Optimization Specialist"

    def test_create_researcher_agent(self):
        from arcane.agents.researcher import create_researcher_agent

        agent = create_researcher_agent()
        assert agent.role == "Deep Research Specialist"
        assert len(agent.tools) >= 4


class TestTaskCreation:
    """Tests for task factory functions."""

    def test_create_planning_task(self):
        from arcane.agents.crew import create_planning_task

        task = create_planning_task("What is quantum computing?")
        assert "quantum computing" in task.description
        assert task.expected_output is not None

    def test_create_query_generation_task(self):
        from arcane.agents.crew import create_query_generation_task

        task = create_query_generation_task(["Q1?", "Q2?"])
        assert "Q1?" in task.description
        assert "Q2?" in task.description

    def test_create_research_task(self):
        from arcane.agents.crew import create_research_task

        task = create_research_task("protein folding AI")
        assert "protein folding" in task.description

    def test_create_synthesis_task(self):
        from arcane.agents.crew import create_synthesis_task

        task = create_synthesis_task("test query", "findings data")
        assert "test query" in task.description
        assert "findings data" in task.description

    def test_create_synthesis_task_with_feedback(self):
        from arcane.agents.crew import create_synthesis_task

        task = create_synthesis_task("query", "findings", "needs more citations")
        assert "REVIEWER FEEDBACK" in task.description
        assert "needs more citations" in task.description

    def test_create_critique_task(self):
        from arcane.agents.crew import create_critique_task

        task = create_critique_task("Sample report text", "original query")
        assert "original query" in task.description
        assert "Source Credibility" in task.description


class TestCrewAssembly:
    """Tests for crew assembly functions."""

    def test_assemble_planning_crew(self):
        from arcane.agents.crew import assemble_planning_crew

        crew = assemble_planning_crew("test query")
        assert len(crew.agents) == 1
        assert len(crew.tasks) == 1

    def test_assemble_critique_crew(self):
        from arcane.agents.crew import assemble_critique_crew

        crew = assemble_critique_crew("report text", "query")
        assert len(crew.agents) == 1
        assert len(crew.tasks) == 1
