
# main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
from app.services.career_analysis_service import CareerAnalysisService
from app.schemas.analysis import AnalysisRequest, AnalysisResponse
import uvicorn

app = FastAPI(title="Career Gap Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the service
career_service = CareerAnalysisService()

@app.post("/analyze-career-gap", response_model=AnalysisResponse)
async def analyze_career_gap(
    file: UploadFile = File(...),
    target_role: str = Form(default="Software Engineer")
):
    """
    Analyze career gap between student resume and industry requirements
    """
    try:
        # Validate file type
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Run the analysis
            result = await career_service.analyze_career_gap(temp_file_path, target_role)
            return result
        finally:
            # Clean up temp file
            os.unlink(temp_file_path)
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Career Gap Analysis API is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
