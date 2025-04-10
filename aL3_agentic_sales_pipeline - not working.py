# Warning control
import warnings
warnings.filterwarnings('ignore')

import asyncio
import os
import json
import yaml
import pandas as pd
import textwrap
from crewai import Agent, Task, Crew, LLM, Flow
from crewai.flow.flow import listen, start
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from pydantic import BaseModel, Field, ConfigDict
from typing import Dict, Optional, List, Set, Tuple

# Environment setup
openai_api_key = os.environ.get('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o-mini'

"""
# OpenRouter LLM setup
openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")
openrouter_model_name = "meta-llama/llama-4-maverick:free"
print(f"--- Configuring LLM for OpenRouter: {openrouter_model_name} ---")
openrouter_llm = LLM(
    model=openrouter_model_name,
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
    max_tokens=1000,
    temperature=0.7
)
"""

# Load YAML configurations
files = {
    'lead_agents': 'config/lead_qualification_agents.yaml',
    'lead_tasks': 'config/lead_qualification_tasks.yaml',
    'email_agents': 'config/email_engagement_agents.yaml',
    'email_tasks': 'config/email_engagement_tasks.yaml'
}
configs = {}
for config_type, file_path in files.items():
    with open(file_path, 'r') as file:
        configs[config_type] = yaml.safe_load(file)

lead_agents_config = configs['lead_agents']
lead_tasks_config = configs['lead_tasks']
email_agents_config = configs['email_agents']
email_tasks_config = configs['email_tasks']

# Pydantic models with ConfigDict to address deprecation warning
class LeadPersonalInfo(BaseModel):
    name: str = Field(..., description="The full name of the lead.")
    job_title: str = Field(..., description="The job title of the lead.")
    role_relevance: int = Field(..., ge=0, le=10, description="Role relevance score (0-10).")
    professional_background: Optional[str] = Field(..., description="Lead's professional background.")
    model_config = ConfigDict(arbitrary_types_allowed=True)

class CompanyInfo(BaseModel):
    company_name: str = Field(..., description="Company name.")
    industry: str = Field(..., description="Company industry.")
    company_size: int = Field(..., description="Employee count.")
    revenue: Optional[float] = Field(None, description="Annual revenue, if available.")
    market_presence: int = Field(..., ge=0, le=10, description="Market presence score (0-10).")
    model_config = ConfigDict(arbitrary_types_allowed=True)

class LeadScore(BaseModel):
    score: int = Field(..., ge=0, le=100, description="Final lead score (0-100).")
    scoring_criteria: List[str] = Field(..., description="Scoring criteria.")
    validation_notes: Optional[str] = Field(None, description="Validation notes.")
    model_config = ConfigDict(arbitrary_types_allowed=True)

class LeadScoringResult(BaseModel):
    personal_info: LeadPersonalInfo = Field(..., description="Personal info.")
    company_info: CompanyInfo = Field(..., description="Company info.")
    lead_score: LeadScore = Field(..., description="Lead score info.")
    model_config = ConfigDict(arbitrary_types_allowed=True)

# Lead Scoring Crew
lead_data_agent = Agent(config=lead_agents_config['lead_data_agent'], tools=[SerperDevTool(), ScrapeWebsiteTool()])
cultural_fit_agent = Agent(config=lead_agents_config['cultural_fit_agent'], tools=[SerperDevTool(), ScrapeWebsiteTool()])
scoring_validation_agent = Agent(config=lead_agents_config['scoring_validation_agent'], tools=[SerperDevTool(), ScrapeWebsiteTool()])

lead_data_task = Task(config=lead_tasks_config['lead_data_collection'], agent=lead_data_agent)
cultural_fit_task = Task(config=lead_tasks_config['cultural_fit_analysis'], agent=cultural_fit_agent)
scoring_validation_task = Task(
    config=lead_tasks_config['lead_scoring_and_validation'],
    agent=scoring_validation_agent,
    context=[lead_data_task, cultural_fit_task],
    output_pydantic=LeadScoringResult
)

lead_scoring_crew = Crew(
    agents=[lead_data_agent, cultural_fit_agent, scoring_validation_agent],
    tasks=[lead_data_task, cultural_fit_task, scoring_validation_task],
    verbose=True
)

# Email Writing Crew
email_content_specialist = Agent(config=email_agents_config['email_content_specialist'])
engagement_strategist = Agent(config=email_agents_config['engagement_strategist'])

email_drafting = Task(config=email_tasks_config['email_drafting'], agent=email_content_specialist)
engagement_optimization = Task(config=email_tasks_config['engagement_optimization'], agent=engagement_strategist)

email_writing_crew = Crew(
    agents=[email_content_specialist, engagement_strategist],
    tasks=[email_drafting, engagement_optimization],
    verbose=True
)

# Sales Pipeline Flow
class SalesPipeline(Flow):
    @start()
    def fetch_leads(self):
        leads = [{"lead_data": {
            "name": "João Moura",
            "job_title": "Director of Engineering",
            "company": "Clearbit",
            "email": "joao@clearbit.com",
            "use_case": "Using AI Agent to do better data enrichment."
        }}]
        return leads

    @listen(fetch_leads)
    def score_leads(self, leads):
        scores = lead_scoring_crew.kickoff_for_each(leads)
        self.state["score_crews_results"] = scores
        return scores

    @listen(score_leads)
    def store_leads_score(self, scores):
        return scores

    @listen(score_leads)
    def filter_leads(self, scores):
        return [score for score in scores if score['lead_score'].score > 70]

    @listen(filter_leads)
    def write_email(self, leads):
        scored_leads = [lead.to_dict() for lead in leads]
        emails = email_writing_crew.kickoff_for_each(scored_leads)
        return emails

    @listen(write_email)
    def send_email(self, emails):
        return emails

# Main execution function
async def main():
    flow = SalesPipeline()
    flow.plot()  # Generates crewai_flow.html

    # Kick off the flow
    print("--- Starting Sales Pipeline ---")
    emails = await flow.kickoff()

    # Token usage from scoring
    df_usage_metrics_scores = pd.DataFrame([flow.state["score_crews_results"][0].token_usage.dict()])
    costs_scores = 0.150 * df_usage_metrics_scores['total_tokens'].sum() / 1_000_000
    print(f"\nTotal costs (scoring): ${costs_scores:.4f}")
    print("Scoring Token Usage Metrics:")
    print(df_usage_metrics_scores.to_string(index=False))

    # Token usage from emails
    df_usage_metrics_emails = pd.DataFrame([emails[0].token_usage.dict()])
    costs_emails = 0.150 * df_usage_metrics_emails['total_tokens'].sum() / 1_000_000
    print(f"\nTotal costs (emails): ${costs_emails:.4f}")
    print("Email Token Usage Metrics:")
    print(df_usage_metrics_emails.to_string(index=False))

    # Lead scoring results
    scores = flow.state["score_crews_results"]
    lead_scoring_result = scores[0].pydantic
    data = {
        'Name': lead_scoring_result.personal_info.name,
        'Job Title': lead_scoring_result.personal_info.job_title,
        'Role Relevance': lead_scoring_result.personal_info.role_relevance,
        'Professional Background': lead_scoring_result.personal_info.professional_background,
        'Company Name': lead_scoring_result.company_info.company_name,
        'Industry': lead_scoring_result.company_info.industry,
        'Company Size': lead_scoring_result.company_info.company_size,
        'Revenue': lead_scoring_result.company_info.revenue,
        'Market Presence': lead_scoring_result.company_info.market_presence,
        'Lead Score': lead_scoring_result.lead_score.score,
        'Scoring Criteria': ', '.join(lead_scoring_result.lead_score.scoring_criteria),
        'Validation Notes': lead_scoring_result.lead_score.validation_notes
    }
    df = pd.DataFrame(list(data.items()), columns=['Attribute', 'Value'])
    print("\nLead Scoring Results:")
    for _, row in df.iterrows():
        print(f"{row['Attribute']:<20}: {row['Value']}")

    # Email result
    print("\nEmail Result:")
    result_text = emails[0].raw
    wrapped_text = textwrap.fill(result_text, width=80)
    print(wrapped_text)

# Robust event loop handling
def run_main():
    # Get or create an event loop in a way that avoids conflicts
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # If no loop is running, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Run the main coroutine
    if loop.is_running():
        # If a loop is already running, use it directly (e.g., in a notebook)
        return loop.run_until_complete(main())
    else:
        # If no loop is running, run it normally
        return loop.run_until_complete(main())

if __name__ == "__main__":
    run_main()