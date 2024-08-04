import markdown
from fastapi import HTTPException
from fastapi import FastAPI, File, Form, UploadFile, Depends, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import google.generativeai as genai
import markdown2
import os
import requests
from dotenv import load_dotenv

# Store these pdfs in a database, user_id, 

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# load environmental variables
load_dotenv()

origins = ["http://127.0.0.1:8000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


API_KEY = os.getenv('GEMINI_API_KEY')

# Assuming genai.configure expects keyword arguments
genai.configure(api_key=API_KEY)

# Set up the model configuration
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

# Initialize the model
model = genai.GenerativeModel(
  model_name="tunedModels/brightspend-ai-training-ilpn6zzcubfi",
  generation_config=generation_config,
  # safety_settings = Adjust safety settings
  # See https://ai.google.dev/gemini-api/docs/safety-settings
)

current_image_path = None

@app.get("/text")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})  # Assuming your template is named "index.html"

@app.post("/query")
async def handle_input(request: Request, user_input):
    response = model.generate_content(user_input)
    response_data = {"response": response.text}
    return JSONResponse(content=response_data, status_code=201)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/favicon.ico")
async def favicon():
    pass

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global current_image_path
    img_path = Path(__file__).resolve().parent / "uploads" / file.filename
    img_path.parent.mkdir(parents=True, exist_ok=True)
    with img_path.open("wb") as buffer:
        buffer.write(file.file.read())
    current_image_path = img_path
    print('uploaded')

def to_html(markdown_format):
    return (
        markdown2.markdown(markdown_format)
        .replace("\\", "")
        .replace("<h1>", "<h7>")
        .replace("</h1>", "</h7>")
        .replace("\\\\", "")
        .replace("```", "")
        .replace("python", "")
        .replace("\n","<br>")
        .replace('"',"")
        .replace("#","<b>")
    )

def removeEmpty(paragraph):
    lines = paragraph.split('\n')
    non_empty_lines = [line for line in lines if line.strip() != '']
    cleaned_paragraph = '\n'.join(non_empty_lines)
    return cleaned_paragraph

"""
@app.post("/gemini")
async def query(query: str, model_type: str = Query(default='text')):
    if not query:
        return ''
    models = {'text': model}
    model = models.get(model_type)

    if not model:
        raise HTTPException(status_code=400, detail="Invalid model type")

    response = model.generate_content(query)
    return JSONResponse(content=response.text)
"""
@app.get("/gemini/img")
async def queryimg(query: str, model_type: str = Query(default='image')):
    global current_image_path
    img_data = current_image_path.read_bytes()
    img = [{"mime_type": "image/jpeg", "data": img_data}]

    # model = generative_text_model if model_type == 'text' else generative_image_model

    if model_type not in {'text', 'image'}:
        raise HTTPException(status_code=400, detail="Invalid model type")

    # response = model.generate_content([query, img[0]], stream=True)
    # response.resolve()
    # current_image_path = None

   # return removeEmpty(to_html(response.text))
