from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_ai import Agent

# 1. Define the structured schema
class SkillInfo(BaseModel):
    skill_name: str
    years_experience: Optional[int] = None

class CandidateProfile(BaseModel):
    name: str
    current_role: str
    skills: List[SkillInfo]

# 2. Update 'result_type' to 'output_type'
profile_agent = Agent(
    model='google:gemini-3.7-flash', 
    output_type=CandidateProfile,  # <-- Change this from result_type
    system_instruction=(
        "You are an expert HR data parsing assistant. "
        "Extract professional details from the text into the requested structure."
    )
)

raw_resume_snippet = """
Hey team, I'm Alex Mercer. Currently I've been working as a Senior DevOps Engineer here. 
I have spent the last 4 years heavily writing Python scripts and automating deployments.
"""

def run_synchronous_demo():
    result = profile_agent.run_sync(user_prompt=raw_resume_snippet)
    
    # 3. Update 'result.data' to 'result.output'
    profile: CandidateProfile = result.output  # <-- Change this from result.data
    print(f"Candidate Name: {profile.name}")
    print(f"Role: {profile.current_role}")

if __name__ == "__main__":
    run_synchronous_demo()
