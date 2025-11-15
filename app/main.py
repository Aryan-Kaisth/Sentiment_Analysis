from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from app.schemas import SentimentText
from src.pipelines.prediction_pipeline import PredictionPipeline

app = FastAPI(title="Sentiment Analyzer")
pipeline = PredictionPipeline()

# Serve static files (CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve HTML templates
templates = Jinja2Templates(directory="templates")


# FRONTEND ROUTE
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, user_text: str = Form(...)):

    # Validate input using Pydantic
    input_data = SentimentText(Text=user_text)

    # Get raw model output (NumPy array)
    preds = pipeline.predict(input_data.Text)  # shape (1,1)

    # Convert raw prediction to sentiment and probability
    probability = float(preds[0][0])
    sentiment = "positive" if probability >= 0.5 else "negative"

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "tweet": input_data.Text,
            "prediction": sentiment
        }
    )
