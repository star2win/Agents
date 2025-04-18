# Warning control
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew

import os
import yaml

openai_api_key = os.environ.get('OPENAI_API_KEY')
if not openai_api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o-mini'

from crewai import Agent, Task, Crew

# Define file paths for YAML configurations
files = {
    'agents': './config/agents.yaml',
    'tasks': './config/tasks.yaml'
}

# Load configurations from YAML files
configs = {}
for config_type, file_path in files.items():
    with open(file_path, 'r') as file:
        configs[config_type] = yaml.safe_load(file)

# Assign loaded configurations to specific variables
agents_config = configs['agents']
tasks_config = configs['tasks']

from typing import List
from pydantic import BaseModel, Field

class TaskEstimate(BaseModel):
    task_name: str = Field(..., description="Name of the task")
    estimated_time_hours: float = Field(..., description="Estimated time to complete the task in hours")
    required_resources: List[str] = Field(..., description="List of resources required to complete the task")

class Milestone(BaseModel):
    milestone_name: str = Field(..., description="Name of the milestone")
    tasks: List[str] = Field(..., description="List of task IDs associated with this milestone")

class ProjectPlan(BaseModel):
    tasks: List[TaskEstimate] = Field(..., description="List of tasks with their estimates")
    milestones: List[Milestone] = Field(..., description="List of project milestones")

# Creating Agents
project_planning_agent = Agent(
  config=agents_config['project_planning_agent']
)

estimation_agent = Agent(
  config=agents_config['estimation_agent']
)

resource_allocation_agent = Agent(
  config=agents_config['resource_allocation_agent']
)

# Creating Tasks
task_breakdown = Task(
  config=tasks_config['task_breakdown'],
  agent=project_planning_agent
)

time_resource_estimation = Task(
  config=tasks_config['time_resource_estimation'],
  agent=estimation_agent
)

resource_allocation = Task(
  config=tasks_config['resource_allocation'],
  agent=resource_allocation_agent,
  output_pydantic=ProjectPlan # This is the structured output we want
)

# Creating Crew
crew = Crew(
  agents=[
    project_planning_agent,
    estimation_agent,
    resource_allocation_agent
  ],
  tasks=[
    task_breakdown,
    time_resource_estimation,
    resource_allocation
  ],
  verbose=True
)

project = 'Auto Shop Website'
industry = 'Technology'
project_objectives = 'Create a website for an auto shop business'
team_members = """
- John Doe (Project Manager)
- Jane Doe (Software Engineer)
- Bob Smith (Designer)
- Alice Johnson (QA Engineer)
- Tom Brown (QA Engineer)
"""
project_requirements = """
- Create a responsive design that works well on desktop and mobile devices
- Implement a modern, visually appealing user interface with a clean look
- Develop a user-friendly navigation system with intuitive menu structure
- Include an "About Us" page highlighting the company's history and values
- Design a "Services" page showcasing the business's automotive offerings with descriptions
- Create a "Contact Us" page with a form and integrated map for communication
- Implement a blog section for sharing industry news and company updates
- Ensure fast loading times and optimize for search engines (SEO)
- Integrate social media links and sharing capabilities
- Include a testimonials section to showcase customer feedback and build trust
"""

from rich.markdown import Markdown as RichMarkdown
from rich.console import Console
from rich.table import Table

# Format the dictionary as Markdown for a better display in Jupyter Lab
formatted_output = f"""
**Project Type:** {project}

**Project Objectives:** {project_objectives}

**Industry:** {industry}

**Team Members:**
{team_members}
**Project Requirements:**
{project_requirements}
"""

# Render the Markdown in the terminal
console = Console()
console.print(RichMarkdown(formatted_output))

# The given Python dictionary
inputs = {
  'project_type': project,
  'project_objectives': project_objectives,
  'industry': industry,
  'team_members': team_members,
  'project_requirements': project_requirements
}

# Run the crew
result = crew.kickoff(
  inputs=inputs
)

import pandas as pd

costs = 0.150 * (crew.usage_metrics.prompt_tokens + crew.usage_metrics.completion_tokens) / 1_000_000
print(f"Total costs: ${costs:.4f}")

# Initialize rich console
console = Console()

# Convert UsageMetrics instance to a DataFrame
df_usage_metrics = pd.DataFrame([crew.usage_metrics.model_dump()])
# Display usage metrics as a rich table
console.print("\n[bold]Usage Metrics:[/bold]")
table = Table(show_header=True, header_style="bold magenta")
for col in df_usage_metrics.columns:
    table.add_column(str(col))
for row in df_usage_metrics.itertuples(index=False):
    table.add_row(*[str(cell) for cell in row])
console.print(table)

# Extract tasks and milestones
tasks = result.pydantic.model_dump()['tasks']
df_tasks = pd.DataFrame(tasks)

# Display tasks as a rich table
console.print("\n[bold]Task Details:[/bold]")
table = Table(show_header=True, header_style="bold magenta")
for col in df_tasks.columns:
    table.add_column(str(col))
for row in df_tasks.itertuples(index=False):
    table.add_row(*[str(cell) for cell in row])
console.print(table)

milestones = result.pydantic.model_dump()['milestones']
df_milestones = pd.DataFrame(milestones)

# Display milestones as a rich table
console.print("\n[bold]Milestones:[/bold]")
table = Table(show_header=True, header_style="bold magenta")
for col in df_milestones.columns:
    table.add_column(str(col))
for row in df_milestones.itertuples(index=False):
    table.add_row(*[str(cell) for cell in row])
console.print(table)

# Optionally, save the styled DataFrames as HTML files
styled_tasks = df_tasks.style.set_table_attributes('border="1"').set_caption("Task Details").set_table_styles(
    [{'selector': 'th, td', 'props': [('font-size', '120%')]}]
)
styled_milestones = df_milestones.style.set_table_attributes('border="1"').set_caption("Task Details").set_table_styles(
    [{'selector': 'th, td', 'props': [('font-size', '120%')]}]
)

# Save to HTML files
styled_tasks.to_html("tasks.html")
styled_milestones.to_html("milestones.html")
print("\nStyled tables have been saved as 'tasks.html' and 'milestones.html'. Open them in a browser to view.")