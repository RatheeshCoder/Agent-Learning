from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class AnalysisRequest(BaseModel):
    target_role: str = "Software Engineer"

class StudentProfile(BaseModel):
    skills: List[str]
    education: str
    experience: str  
    projects: str

class AnalysisResponse(BaseModel):
    student_profile: StudentProfile
    alignment_score: int
    matched_skills: List[str]
    missing_skills: List[str]
    industry_requirements: str
    recommendations: Dict[str, List[str]]
    analysis_summary: str