import os
import PyPDF2
import google.generativeai as genai
from tavily import TavilyClient
from typing import List, Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
import re
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import numpy as np
from datetime import datetime
import hashlib
import json

from app.schemas.analysis import AnalysisResponse, StudentProfile

# Configuration
os.environ["GOOGLE_API_KEY"] = "AIzaSyB_KtLx3UmzQKeC4myIMej7A0Rsh_aS_CY"
os.environ["TAVILY_API_KEY"] = "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"

class AgentState(TypedDict):
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

class RAGResumeProcessor:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/", db_name: str = "career_analysis"):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db.resume_vectors
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create vector search index if it doesn't exist
        self._create_vector_index()
    
    def _create_vector_index(self):
        """Create vector search index in MongoDB"""
        try:
            # Check if index exists
            indexes = list(self.collection.list_indexes())
            vector_index_exists = any(idx.get('name') == 'vector_index' for idx in indexes)
            
            if not vector_index_exists:
                self.collection.create_index([
                    ("embedding", "2dsphere")
                ], name="vector_index")
                print("✅ Vector index created")
        except Exception as e:
            print(f"Index creation note: {e}")
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def _generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for text chunks"""
        embeddings = self.embedding_model.encode(texts)
        return embeddings.tolist()
    
    def _get_pdf_hash(self, pdf_path: str) -> str:
        """Generate hash for PDF file to check if already processed"""
        with open(pdf_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash
    
    def process_and_store_resume(self, pdf_path: str, pdf_text: str) -> str:
        """Process resume PDF and store in MongoDB with embeddings"""
        try:
            pdf_hash = self._get_pdf_hash(pdf_path)
            
            # Check if already processed
            existing = self.collection.find_one({"pdf_hash": pdf_hash})
            if existing:
                print("📋 Resume already processed, using existing data")
                return pdf_hash
            
            # Chunk the text
            chunks = self._chunk_text(pdf_text)
            print(f"📄 Created {len(chunks)} chunks from resume")
            
            # Generate embeddings
            embeddings = self._generate_embeddings(chunks)
            
            # Store in MongoDB
            documents = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                doc = {
                    "pdf_hash": pdf_hash,
                    "pdf_path": pdf_path,
                    "chunk_id": i,
                    "text": chunk,
                    "embedding": embedding,
                    "created_at": datetime.now(),
                    "metadata": {
                        "chunk_size": len(chunk),
                        "word_count": len(chunk.split())
                    }
                }
                documents.append(doc)
            
            # Insert documents
            result = self.collection.insert_many(documents)
            print(f"✅ Stored {len(result.inserted_ids)} resume chunks in MongoDB")
            
            return pdf_hash
            
        except Exception as e:
            print(f"Error processing resume: {e}")
            return None
    
    def semantic_search(self, query: str, pdf_hash: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks using semantic similarity"""
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Get all chunks for this PDF
            chunks = list(self.collection.find({"pdf_hash": pdf_hash}))
            
            if not chunks:
                return []
            
            # Calculate similarities
            similarities = []
            for chunk in chunks:
                chunk_embedding = chunk["embedding"]
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                )
                similarities.append({
                    "chunk": chunk,
                    "similarity": similarity
                })
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            print(f"Error in semantic search: {e}")
            return []

class CareerAnalysisService:
    def __init__(self, mongo_uri: str = "mongodb://localhost:27017/"):
        self.tavily_client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.gemini_model = genai.GenerativeModel("gemini-1.5-flash")
        self.rag_processor = RAGResumeProcessor(mongo_uri)
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
    
    def _extract_student_info_with_rag(self, pdf_hash: str) -> Dict[str, Any]:
        """Extract skills, education, experience using RAG system"""
        try:
            # Define search queries for different sections
            search_queries = {
                "skills": "technical skills programming languages frameworks tools technologies software development",
                "education": "education degree university college school academic qualification",
                "experience": "work experience job employment internship professional career",
                "projects": "projects portfolio development built created implemented"
            }
            
            extracted_info = {}
            
            for section, query in search_queries.items():
                # Get relevant chunks
                relevant_chunks = self.rag_processor.semantic_search(query, pdf_hash, top_k=3)
                
                if relevant_chunks:
                    # Combine relevant text
                    combined_text = "\n".join([chunk["chunk"]["text"] for chunk in relevant_chunks])
                    
                    # Use Gemini to extract specific information
                    if section == "skills":
                        prompt = f"""
                        Extract all technical skills, programming languages, frameworks, tools, and technologies from the following text.
                        Return only a JSON list of skills, no explanations.
                        
                        Text: {combined_text}
                        
                        Format: ["skill1", "skill2", "skill3", ...]
                        """
                    elif section == "education":
                        prompt = f"""
                        Extract education information including degree, institution, year, GPA from the following text.
                        Return a concise summary in 1-2 sentences.
                        
                        Text: {combined_text}
                        """
                    elif section == "experience":
                        prompt = f"""
                        Extract work experience information including company names, positions, duration from the following text.
                        Return a concise summary in 2-3 sentences.
                        
                        Text: {combined_text}
                        """
                    else:  # projects
                        prompt = f"""
                        Extract project information including project names, technologies used, descriptions from the following text.
                        Return a concise summary in 2-3 sentences.
                        
                        Text: {combined_text}
                        """
                    
                    response = self.gemini_model.generate_content(prompt)
                    
                    if section == "skills":
                        try:
                            # Try to parse JSON response
                            skills_text = response.text.strip()
                            # Clean up the response to extract JSON
                            if skills_text.startswith('```json'):
                                skills_text = skills_text.replace('```json', '').replace('```', '').strip()
                            elif skills_text.startswith('```'):
                                skills_text = skills_text.replace('```', '').strip()
                            
                            skills = json.loads(skills_text)
                            extracted_info[section] = skills if isinstance(skills, list) else []
                        except:
                            # Fallback: extract skills using regex
                            skills_text = response.text
                            # Try to find skills in various formats
                            skills = re.findall(r'"([^"]+)"', skills_text)
                            if not skills:
                                skills = [s.strip() for s in skills_text.split(',') if s.strip()]
                            extracted_info[section] = skills[:20]  # Limit to 20 skills
                    else:
                        extracted_info[section] = response.text.strip()[:300]  # Limit length
                else:
                    extracted_info[section] = "Not specified" if section != "skills" else []
            
            print(f"✅ RAG extraction completed: {len(extracted_info.get('skills', []))} skills found")
            return extracted_info
            
        except Exception as e:
            print(f"Error in RAG extraction: {e}")
            return {
                "skills": [],
                "education": "Not specified",
                "experience": "Not specified",
                "projects": "Not specified"
            }

    def pdf_extraction_agent(self, state: AgentState) -> AgentState:
        """Agent 1: Extract data from student resume PDF using RAG"""
        print("🔍 Agent 1: Extracting data from PDF using RAG system...")
        
        pdf_text = self._extract_text_from_pdf(state["pdf_path"])
        
        if not pdf_text:
            print("⚠️ Could not extract text from PDF, using fallback data")
            student_info = {
                "skills": ["Python", "JavaScript", "React"],
                "education": "Bachelor of Computer Science",
                "experience": "Internship experience",
                "projects": "Web development projects"
            }
        else:
            # Process and store in MongoDB with RAG
            pdf_hash = self.rag_processor.process_and_store_resume(state["pdf_path"], pdf_text)
            
            if pdf_hash:
                # Extract information using RAG
                student_info = self._extract_student_info_with_rag(pdf_hash)
            else:
                print("⚠️ RAG processing failed, using fallback extraction")
                # Fallback to basic extraction
                student_info = self._basic_extraction_fallback(pdf_text)
        
        print(f"✅ Extracted {len(student_info['skills'])} skills: {student_info['skills'][:10]}...")
        
        return {
            "pdf_path": state["pdf_path"],
            "target_role": state["target_role"],
            "student_skills": student_info["skills"],
            "student_education": student_info["education"],
            "student_experience": student_info["experience"],
            "student_projects": student_info["projects"],
            "industry_requirements": "",
            "industry_skills": [],
            "matched_skills": [],
            "missing_skills": [],
            "alignment_score": 0,
            "recommendations": {}
        }
    
    def _basic_extraction_fallback(self, text: str) -> Dict[str, Any]:
        """Fallback extraction method when RAG fails"""
        try:
            prompt = f"""
            Extract information from this resume text:
            
            {text[:2000]}
            
            Return a JSON object with:
            - skills: array of technical skills
            - education: education summary
            - experience: work experience summary  
            - projects: projects summary
            
            Format as valid JSON only, no explanations.
            """
            
            response = self.gemini_model.generate_content(prompt)
            result = json.loads(response.text.strip())
            return result
        except:
            return {
                "skills": ["Python", "JavaScript", "React"],
                "education": "Computer Science background",
                "experience": "Software development experience",
                "projects": "Various development projects"
            }

    def skill_matching_agent(self, state: AgentState) -> AgentState:
        """Agent 2: Get industry requirements and match skills"""
        print("🌐 Agent 2: Fetching industry requirements and matching skills...")
        
        try:
            search_query = f"{state['target_role']} skills requirements 2024 hiring"
            search_results = self.tavily_client.search(search_query, max_results=5)
            
            industry_text = ""
            for result in search_results.get("results", []):
                industry_text += result.get("content", "") + " "
            
            prompt = f"""
            Analyze the following industry requirements for {state['target_role']} position:

            {industry_text[:2000]}

            Extract the top 15 technical skills mentioned. Return only a JSON array of skills.
            Format: ["skill1", "skill2", "skill3", ...]
            """
            
            response = self.gemini_model.generate_content(prompt)
            try:
                industry_skills_text = response.text.strip()
                if industry_skills_text.startswith('```json'):
                    industry_skills_text = industry_skills_text.replace('```json', '').replace('```', '').strip()
                elif industry_skills_text.startswith('```'):
                    industry_skills_text = industry_skills_text.replace('```', '').strip()
                
                industry_skills = json.loads(industry_skills_text)
                if not isinstance(industry_skills, list):
                    raise ValueError("Not a list")
            except:
                # Fallback parsing
                industry_skills = [skill.strip() for skill in response.text.split(",")][:15]
            
            # Advanced skill matching with fuzzy matching
            matched_skills = []
            missing_skills = []
            
            for industry_skill in industry_skills:
                found = False
                for student_skill in state["student_skills"]:
                    # Direct match or partial match
                    if (industry_skill.lower() == student_skill.lower() or 
                        industry_skill.lower() in student_skill.lower() or 
                        student_skill.lower() in industry_skill.lower() or
                        self._fuzzy_match_skills(industry_skill, student_skill)):
                        matched_skills.append(student_skill)
                        found = True
                        break
                
                if not found:
                    missing_skills.append(industry_skill)
            
            # Remove duplicates
            matched_skills = list(dict.fromkeys(matched_skills))
            
            alignment_score = int((len(matched_skills) / len(industry_skills)) * 100) if industry_skills else 0
            
            print(f"✅ Found {len(industry_skills)} industry skills")
            print(f"✅ Matched {len(matched_skills)} skills: {matched_skills[:5]}...")
            print(f"✅ Missing {len(missing_skills)} skills: {missing_skills[:5]}...")
            print(f"✅ Alignment Score: {alignment_score}%")
        
        except Exception as e:
            print(f"Error in skill matching: {e}")
            industry_skills = ["Python", "JavaScript", "React", "Node.js", "Docker"]
            matched_skills = [skill for skill in state["student_skills"] if skill.lower() in [s.lower() for s in industry_skills]]
            missing_skills = [skill for skill in industry_skills if skill.lower() not in [s.lower() for s in state["student_skills"]]]
            alignment_score = 50
            industry_text = "Error fetching industry requirements"
        
        return {
            "pdf_path": state["pdf_path"],
            "target_role": state["target_role"],
            "student_skills": state["student_skills"],
            "student_education": state["student_education"],
            "student_experience": state["student_experience"],
            "student_projects": state["student_projects"],
            "industry_requirements": industry_text[:500],
            "industry_skills": industry_skills,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "alignment_score": alignment_score,
            "recommendations": {}
        }
    
    def _fuzzy_match_skills(self, skill1: str, skill2: str, threshold: float = 0.8) -> bool:
        """Simple fuzzy matching for skills"""
        # Remove common suffixes/prefixes
        clean_skill1 = skill1.lower().replace('.js', '').replace('js', '').strip()
        clean_skill2 = skill2.lower().replace('.js', '').replace('js', '').strip()
        
        # Check if one is contained in another with good ratio
        if clean_skill1 in clean_skill2 or clean_skill2 in clean_skill1:
            return True
            
        return False

    def recommendation_agent(self, state: AgentState) -> AgentState:
        """Agent 3: Generate learning recommendations"""
        print("💡 Agent 3: Generating learning recommendations...")
        
        try:
            prompt = f"""
            Student Profile:
            - Current Skills: {', '.join(state['student_skills'])}
            - Target Role: {state['target_role']}
            - Missing Skills: {', '.join(state['missing_skills'][:10])}
            - Alignment Score: {state['alignment_score']}%

            Generate specific learning recommendations as JSON with simple string arrays:
            {{
                "priority_skills": ["skill1", "skill2", "skill3", "skill4", "skill5"],
                "online_courses": ["course title 1", "course title 2", "course title 3"],
                "projects": ["project idea 1", "project idea 2", "project idea 3"],
                "certifications": ["certification 1", "certification 2"]
            }}

            Make sure all values are simple strings, not objects. Keep recommendations practical and actionable.
            Example format:
            {{
                "priority_skills": ["Python", "AWS", "Docker"],
                "online_courses": ["Complete Python Bootcamp on Udemy", "AWS Solutions Architect Course", "Docker Mastery Course"],
                "projects": ["Build a web scraper with Python", "Deploy app on AWS", "Containerize application with Docker"],
                "certifications": ["AWS Solutions Architect", "Docker Certified Associate"]
            }}
            """
            
            response = self.gemini_model.generate_content(prompt)
            
            try:
                rec_text = response.text.strip()
                if rec_text.startswith('```json'):
                    rec_text = rec_text.replace('```json', '').replace('```', '').strip()
                elif rec_text.startswith('```'):
                    rec_text = rec_text.replace('```', '').strip()
                
                recommendations = json.loads(rec_text)
                
                # Validate and fix the structure to ensure all values are strings
                validated_recommendations = {}
                
                # Handle priority_skills
                if "priority_skills" in recommendations:
                    priority_skills = recommendations["priority_skills"]
                    if isinstance(priority_skills, list):
                        validated_recommendations["priority_skills"] = [
                            str(skill) if isinstance(skill, str) else skill.get('skill', str(skill)) 
                            for skill in priority_skills
                        ][:5]  # Limit to 5
                    else:
                        validated_recommendations["priority_skills"] = state["missing_skills"][:5]
                else:
                    validated_recommendations["priority_skills"] = state["missing_skills"][:5]
                
                # Handle online_courses
                if "online_courses" in recommendations:
                    courses = recommendations["online_courses"]
                    if isinstance(courses, list):
                        validated_recommendations["online_courses"] = []
                        for course in courses:
                            if isinstance(course, str):
                                validated_recommendations["online_courses"].append(course)
                            elif isinstance(course, dict):
                                # Extract course name from dict
                                course_name = course.get('course', course.get('title', course.get('name', str(course))))
                                validated_recommendations["online_courses"].append(str(course_name))
                            else:
                                validated_recommendations["online_courses"].append(str(course))
                    else:
                        validated_recommendations["online_courses"] = [f"Complete {skill} course" for skill in state["missing_skills"][:3]]
                else:
                    validated_recommendations["online_courses"] = [f"Complete {skill} course" for skill in state["missing_skills"][:3]]
                
                # Handle projects
                if "projects" in recommendations:
                    projects = recommendations["projects"]
                    if isinstance(projects, list):
                        validated_recommendations["projects"] = []
                        for project in projects:
                            if isinstance(project, str):
                                validated_recommendations["projects"].append(project)
                            elif isinstance(project, dict):
                                # Extract project description from dict
                                project_desc = project.get('project', project.get('description', project.get('name', str(project))))
                                validated_recommendations["projects"].append(str(project_desc))
                            else:
                                validated_recommendations["projects"].append(str(project))
                    else:
                        validated_recommendations["projects"] = [f"Build {state['target_role'].lower()} project with {skill}" for skill in state["missing_skills"][:3]]
                else:
                    validated_recommendations["projects"] = [f"Build {state['target_role'].lower()} project with {skill}" for skill in state["missing_skills"][:3]]
                
                # Handle certifications
                if "certifications" in recommendations:
                    certs = recommendations["certifications"]
                    if isinstance(certs, list):
                        validated_recommendations["certifications"] = []
                        for cert in certs:
                            if isinstance(cert, str):
                                validated_recommendations["certifications"].append(cert)
                            elif isinstance(cert, dict):
                                # Extract certification name from dict
                                cert_name = cert.get('certification', cert.get('name', cert.get('title', str(cert))))
                                validated_recommendations["certifications"].append(str(cert_name))
                            else:
                                validated_recommendations["certifications"].append(str(cert))
                    else:
                        validated_recommendations["certifications"] = [f"{skill} certification" for skill in state["missing_skills"][:2] if skill in ["AWS", "Docker", "Kubernetes", "Azure", "GCP", "Python", "JavaScript"]]
                else:
                    validated_recommendations["certifications"] = [f"{skill} certification" for skill in state["missing_skills"][:2] if skill in ["AWS", "Docker", "Kubernetes", "Azure", "GCP", "Python", "JavaScript"]]
                
                recommendations = validated_recommendations
                
            except Exception as json_error:
                print(f"Error parsing JSON recommendations: {json_error}")
                # Fallback recommendations with guaranteed string format
                recommendations = {
                    "priority_skills": [str(skill) for skill in state["missing_skills"][:5]],
                    "online_courses": [f"Complete {skill} fundamentals course" for skill in state["missing_skills"][:3]],
                    "projects": [f"Build a {state['target_role'].lower()} project using {skill}" for skill in state["missing_skills"][:3]],
                    "certifications": [f"{skill} Professional Certification" for skill in state["missing_skills"][:2] if skill in ["AWS", "Docker", "Kubernetes", "Azure", "GCP", "Python", "JavaScript"]]
                }
            
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            # Fallback recommendations with guaranteed string format
            recommendations = {
                "priority_skills": [str(skill) for skill in state["missing_skills"][:5]],
                "online_courses": ["Complete relevant online courses", "Take industry-specific training", "Follow structured learning path"],
                "projects": ["Build portfolio projects", "Create real-world applications", "Contribute to open source"],
                "certifications": ["Get industry certifications", "Pursue professional credentials"]
            }
        
        # Final validation to ensure all values are strings
        for key, value_list in recommendations.items():
            if isinstance(value_list, list):
                recommendations[key] = [str(item) for item in value_list if item]  # Convert to string and filter empty values
            else:
                recommendations[key] = []
        
        print(f"✅ Generated recommendations for {len(state['missing_skills'])} missing skills")
        print(f"✅ Priority Skills: {recommendations.get('priority_skills', [])}")
        print(f"✅ Courses: {len(recommendations.get('online_courses', []))}")
        print(f"✅ Projects: {len(recommendations.get('projects', []))}")
        print(f"✅ Certifications: {len(recommendations.get('certifications', []))}")
        
        return {
            "pdf_path": state["pdf_path"],
            "target_role": state["target_role"],
            "student_skills": state["student_skills"],
            "student_education": state["student_education"],
            "student_experience": state["student_experience"],
            "student_projects": state["student_projects"],
            "industry_requirements": state["industry_requirements"],
            "industry_skills": state["industry_skills"],
            "matched_skills": state["matched_skills"],
            "missing_skills": state["missing_skills"],
            "alignment_score": state["alignment_score"],
            "recommendations": recommendations
        }

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
        print(f"🚀 Starting RAG-based career gap analysis for: {target_role}")
        
        initial_state: AgentState = {
            "pdf_path": pdf_path,
            "target_role": target_role,
            "student_skills": [],
            "student_education": "",
            "student_experience": "",
            "student_projects": "",
            "industry_requirements": "",
            "industry_skills": [],
            "matched_skills": [],
            "missing_skills": [],
            "alignment_score": 0,
            "recommendations": {}
        }
        
        final_state = self.graph.invoke(initial_state)
        
        response = AnalysisResponse(
            student_profile=StudentProfile(
                skills=final_state["student_skills"],
                education=final_state["student_education"],
                experience=final_state["student_experience"],
                projects=final_state["student_projects"]
            ),
            alignment_score=final_state["alignment_score"],
            matched_skills=final_state["matched_skills"],
            missing_skills=final_state["missing_skills"],
            industry_requirements=final_state["industry_requirements"],
            recommendations=final_state["recommendations"],
            analysis_summary=f"""
            RAG-based Analysis for {target_role} position:
            - Alignment Score: {final_state["alignment_score"]}%
            - Skills Matched: {len(final_state["matched_skills"])}/{len(final_state["industry_skills"])}
            - Priority Skills to Learn: {', '.join(final_state["missing_skills"][:5])}
            - Resume processed and stored in vector database

            Focus on the recommended priority skills for maximum impact!
            """
        )
        
        print("✅ RAG-based career gap analysis completed!")
        return response

# Additional utility functions for RAG setup
def setup_mongodb_atlas_vector_search():
    """Setup MongoDB Atlas Vector Search (for production)"""
    setup_script = """
    // MongoDB Atlas Vector Search Index Setup
    // Run this in MongoDB Atlas UI or via MongoDB shell
    
    db.resume_vectors.createSearchIndex({
        "name": "vector_index",
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {
                    "type": "vector",
                    "path": "embedding",
                    "numDimensions": 384,
                    "similarity": "cosine"
                }
            ]
        }
    });
    """
    print("MongoDB Atlas Vector Search setup script:")
    print(setup_script)

if __name__ == "__main__":
    # Initialize the service
    service = CareerAnalysisService()
    print("✅ RAG-based Career Analysis Service initialized!")
    
    # Print MongoDB setup info
    setup_mongodb_atlas_vector_search()