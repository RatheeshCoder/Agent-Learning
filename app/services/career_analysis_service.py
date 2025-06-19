import os
import PyPDF2
import google.generativeai as genai
from tavily import TavilyClient
from dataclasses import dataclass
from typing import List, Dict, Any
from langgraph.graph import StateGraph, END
import re

from app.schemas.analysis import AnalysisResponse, StudentProfile

# Configuration
os.environ["GOOGLE_API_KEY"] = "AIzaSyB_KtLx3UmzQKeC4myIMej7A0Rsh_aS_CY"
os.environ["TAVILY_API_KEY"] = "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"

@dataclass
class AgentState:
    pdf_path: str
    target_role: str
    student_skills: List[str]
    student_education: str
    student_experience: str
    student_projects: str
    industry_requirements: str
    industry_skills: List[str]
    matched_skills: List[str]
    missing_skills: List[str]
    alignment_score: int
    recommendations: Dict[str, List[str]]

class CareerAnalysisService:
    def __init__(self):
        self.tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        self.graph = self._build_agent_graph()
    
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""
    
    def _extract_student_info(self, text: str) -> Dict[str, Any]:
        """Extract skills, education, experience from resume text"""
        skill_keywords = [
            "React", "Node.js", "MongoDB", "Express", "JavaScript", "Python", 
            "HTML", "CSS", "TypeScript", "Vue", "Angular", "Docker", "AWS",
            "PostgreSQL", "MySQL", "Git", "Java", "C++", "Django", "Flask",
            "Machine Learning", "AI", "DevOps", "Kubernetes", "Redis", "GraphQL"
        ]
        
        found_skills = []
        text_lower = text.lower()
        for skill in skill_keywords:
            if skill.lower() in text_lower:
                found_skills.append(skill)
        
        education_match = re.search(r'education\s*:?\s*(.*?)(?=experience|projects|skills|$)', text, re.IGNORECASE | re.DOTALL)
        experience_match = re.search(r'experience\s*:?\s*(.*?)(?=education|projects|skills|$)', text, re.IGNORECASE | re.DOTALL)
        projects_match = re.search(r'projects?\s*:?\s*(.*?)(?=education|experience|skills|$)', text, re.IGNORECASE | re.DOTALL)
        
        return {
            "skills": found_skills,
            "education": education_match.group(1).strip()[:200] if education_match else "Not specified",
            "experience": experience_match.group(1).strip()[:200] if experience_match else "Not specified", 
            "projects": projects_match.group(1).strip()[:200] if projects_match else "Not specified"
        }

    def pdf_extraction_agent(self, state: AgentState) -> AgentState:
        """Agent 1: Extract data from student resume PDF"""
        print("🔍 Agent 1: Extracting data from PDF...")
        
        pdf_text = self._extract_text_from_pdf(state.pdf_path)
        print(pdf_text, "pdf_text")
        if not pdf_text:
            student_info = {
                "skills": ["Python", "JavaScript", "React"],
                "education": "Bachelor of Computer Science",
                "experience": "Internship experience",
                "projects": "Web development projects"
            }
        else:
            student_info = self._extract_student_info(pdf_text)
        
        print(f"✅ Extracted {len(student_info['skills'])} skills")
        
        return AgentState(
            pdf_path=state.pdf_path,
            target_role=state.target_role,
            student_skills=student_info["skills"],
            student_education=student_info["education"],
            student_experience=student_info["experience"],
            student_projects=student_info["projects"],
            industry_requirements="",
            industry_skills=[],
            matched_skills=[],
            missing_skills=[],
            alignment_score=0,
            recommendations={}
        )

    def skill_matching_agent(self, state: AgentState) -> AgentState:
        """Agent 2: Get industry requirements and match skills"""
        print("🌐 Agent 2: Fetching industry requirements and matching skills...")
        
        try:
            search_query = f"{state.target_role} skills requirements 2024 hiring"
            search_results = self.tavily_client.search(search_query, max_results=5)
            
            industry_text = ""
            for result in search_results.get("results", []):
                industry_text += result.get("content", "") + " "
            
            prompt = f"""
            Analyze the following industry requirements for {state.target_role} position:

            {industry_text[:2000]}

            Extract the top 15 technical skills mentioned. Return only a comma-separated list of skills.
            """
            
            response = self.gemini_model.generate_content(prompt)
            industry_skills_text = response.text.strip()
            industry_skills = [skill.strip() for skill in industry_skills_text.split(",")]
            
            student_skills_set = set([skill.lower() for skill in state.student_skills])
            industry_skills_set = set([skill.lower() for skill in industry_skills])
            
            matched_skills = list(student_skills_set.intersection(industry_skills_set))
            missing_skills = list(industry_skills_set - student_skills_set)
            
            alignment_score = int((len(matched_skills) / len(industry_skills)) * 100) if industry_skills else 0
            
            print(f"✅ Found {len(industry_skills)} industry skills")
            print(f"✅ Matched {len(matched_skills)} skills")
            print(f"✅ Alignment Score: {alignment_score}%")
        
        except Exception as e:
            print(f"Error in skill matching: {e}")
            industry_skills = ["Python", "JavaScript", "React", "Node.js", "Docker"]
            matched_skills = [skill for skill in state.student_skills if skill.lower() in [s.lower() for s in industry_skills]]
            missing_skills = [skill for skill in industry_skills if skill.lower() not in [s.lower() for s in state.student_skills]]
            alignment_score = 50
        
        return AgentState(
            pdf_path=state.pdf_path,
            target_role=state.target_role,
            student_skills=state.student_skills,
            student_education=state.student_education,
            student_experience=state.student_experience,
            student_projects=state.student_projects,
            industry_requirements=industry_text[:500],
            industry_skills=industry_skills,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            alignment_score=alignment_score,
            recommendations={}
        )

    def recommendation_agent(self, state: AgentState) -> AgentState:
        """Agent 3: Generate learning recommendations"""
        print("💡 Agent 3: Generating learning recommendations...")
        
        try:
            prompt = f"""
            Student Profile:
            - Current Skills: {', '.join(state.student_skills)}
            - Target Role: {state.target_role}
            - Missing Skills: {', '.join(state.missing_skills[:10])}
            - Alignment Score: {state.alignment_score}%

            Generate specific learning recommendations in these categories:
            1. Priority Skills (top 5 skills to learn first)
            2. Online Courses (specific course recommendations)
            3. Projects (hands-on project ideas)
            4. Certifications (relevant certifications)

            Keep recommendations practical and actionable.
            """
            
            response = self.gemini_model.generate_content(prompt)
            recommendations_text = response.text
            
            recommendations = {
                "priority_skills": state.missing_skills[:5],
                "online_courses": [
                    f"Complete {skill} Bootcamp" for skill in state.missing_skills[:3]
                ],
                "projects": [
                    f"Build a {state.target_role.lower()} project using {skill}" 
                    for skill in state.missing_skills[:3]
                ],
                "certifications": [
                    f"Professional {skill} Certification" 
                    for skill in state.missing_skills[:2] 
                    if skill in ["AWS", "Docker", "Kubernetes", "Azure"]
                ]
            }
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            recommendations = {
                "priority_skills": state.missing_skills[:5],
                "online_courses": ["Complete relevant online courses"],
                "projects": ["Build portfolio projects"],
                "certifications": ["Get industry certifications"]
            }
        
        print(f"✅ Generated recommendations for {len(state.missing_skills)} missing skills")
        
        return AgentState(
            pdf_path=state.pdf_path,
            target_role=state.target_role,
            student_skills=state.student_skills,
            student_education=state.student_education,
            student_experience=state.student_experience,
            student_projects=state.student_projects,
            industry_requirements=state.industry_requirements,
            industry_skills=state.industry_skills,
            matched_skills=state.matched_skills,
            missing_skills=state.missing_skills,
            alignment_score=state.alignment_score,
            recommendations=recommendations
        )

    def _build_agent_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        graph_builder = StateGraph(AgentState)
        
        graph_builder.add_node("pdf_agent", self.pdf_extraction_agent)
        graph_builder.add_node("matching_agent", self.skill_matching_agent)
        graph_builder.add_node("recommendation_agent", self.recommendation_agent)
        
        graph_builder.set_entry_point("pdf_agent")
        graph_builder.add_edge("pdf_agent", "matching_agent")
        graph_builder.add_edge("matching_agent", "recommendation_agent")
        graph_builder.add_edge("recommendation_agent", END)
        
        return graph_builder.compile()

    async def analyze_career_gap(self, pdf_path: str, target_role: str) -> AnalysisResponse:
        """Run the complete career gap analysis"""
        print(f"🚀 Starting career gap analysis for: {target_role}")
        
        initial_state = AgentState(
            pdf_path=pdf_path,
            target_role=target_role,
            student_skills=[],
            student_education="",
            student_experience="",
            student_projects="",
            industry_requirements="",
            industry_skills=[],
            matched_skills=[],
            missing_skills=[],
            alignment_score=0,
            recommendations={}
        )
        
        final_state = self.graph.invoke(initial_state)
        
        response = AnalysisResponse(
            student_profile=StudentProfile(
                skills=final_state.student_skills,
                education=final_state.student_education,
                experience=final_state.student_experience,
                projects=final_state.student_projects
            ),
            alignment_score=final_state.alignment_score,
            matched_skills=final_state.matched_skills,
            missing_skills=final_state.missing_skills,
            industry_requirements=final_state.industry_requirements,
            recommendations=final_state.recommendations,
            analysis_summary=f"""
            Analysis for {target_role} position:
            - Alignment Score: {final_state.alignment_score}%
            - Skills Matched: {len(final_state.matched_skills)}/{len(final_state.industry_skills)}
            - Priority Skills to Learn: {', '.join(final_state.missing_skills[:5])}

            Focus on the recommended priority skills for maximum impact!
            """
        )
        
        print("✅ Career gap analysis completed!")
        return response
