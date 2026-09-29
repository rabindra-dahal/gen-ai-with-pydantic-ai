import os
import asyncio
from typing import List, Optional
from pydantic import BaseModel
from pydantic_ai import Agent
from dotenv import load_dotenv  # Import dotenv

# Load environment variables from the .env file automatically
load_dotenv()

# 1. Define the structured schema
class SkillInfo(BaseModel):
    skill_name: str
    years_experience: Optional[int] = None

class CandidateProfile(BaseModel):
    name: str
    current_role: str
    skills: List[SkillInfo]

# 2. Pydantic AI will now safely pick up the GOOGLE_API_KEY loaded above
profile_agent = Agent(
    model='google:gemini-3.5-flash-lite', 
    output_type=CandidateProfile,
    instructions=(
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
    profile: CandidateProfile = result.output
    print("\n--- Parsing Successful! ---")
    print(f"Candidate Name: {profile.name}")
    print(f"Role:           {profile.current_role}")
    print("Skills:")
    for skill in profile.skills:
        print(f"  - {skill.skill_name}")

if __name__ == "__main__":
    run_synchronous_demo()
