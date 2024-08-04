## Google's Gemini API Integration

FastAPI interface to interact with Google's Gemini generative AI models. 

### Key Features:

* **Text Generation:** Generate creative text formats. based on your input query.
* **Image-To-Text Generation:** Generate creative images explaination inspired by your image prompt. 
* **Fast and Easy:** Designed for effortless exploration and integration with other projects.
* **Clean and Documented Code:** Well-organized code with detailed comments for clear understanding.

### Requirements:

* Python 3.7+
* `requirements.txt`

### Getting Started:

1. Clone this repository.
2. Install the required dependencies: `pip install -r requirements.txt`.
3. Set your Google Cloud Platform API key as an environment variable named `GEMINI_API_KEY`.
4. Run the server: `uvicorn GeminiBackend:app --host 0.0.0.0 --port 8000`.
5. Open http://localhost:8000 in your browser.

### API Documentation:
```
https://ai.google.dev/models/gemini
```
#### Text Generation:

* Endpoint: `/gemini`
* Method: GET
* Parameters:
    * `query`: Text prompt for the model to generate (required).
* Example: `http://localhost:8000/gemini?query=Write a poem about a starry night sky`

#### Image-To-Text Generation:

* Endpoint: `/gemini/img`
* Method: GET
* Parameters:
    * `query`: Image prompt for the model to generate Text(required).
    * `image_url`: URL of the reference image (required).
* Example: `http://localhost:8000/gemini/img?query=Explain This Image&image_url=https://example.com/mountain-forest.jpg`

### Contributing:

We welcome contributions! Feel free to fork this repository, make improvements, and submit pull requests. Read the CONTRIBUTING.md file for guidelines.

### License:

This project is licensed under the MIT License. See the LICENSE file for details.

We hope you find Google_Gemini useful and inspiring! 

### Note:

* Use Google Cloud Platform API key.

