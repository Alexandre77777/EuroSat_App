# main.py
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from classifier import EuroSATClassifier
import base64

MODEL_PATH = "models/best_eurosat_model.h5"

app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"], 
    allow_methods=["*"], 
    allow_headers=["*"]
)

# Глобальный классификатор
classifier = None

@app.on_event("startup")
async def startup():
    """Инициализация при запуске"""
    global classifier
    classifier = EuroSATClassifier(MODEL_PATH, patch_size=64, batch_size=4)
    classifier.load_model()
    print("Модель загружена и прогрета")

@app.post("/check_channels/")
async def check_channels(file: UploadFile = File(...)):
    """Проверка количества каналов"""
    contents = await file.read()
    channels = classifier.check_channels(contents)
    return JSONResponse({"channels": channels})

@app.post("/get_preview/")
async def get_preview(
    file: UploadFile = File(...), 
    channels: str = Query(None, description="Каналы через запятую")
):
    """Получение превью с выбранными каналами"""
    contents = await file.read()
    
    try:
        preview_bytes = classifier.create_preview(contents, channels)
        preview_base64 = base64.b64encode(preview_bytes).decode('utf-8')
        return JSONResponse({"preview": preview_base64, "status": "success"})
    except Exception as e:
        return JSONResponse(
            {"error": f"Ошибка создания превью: {str(e)}", "status": "error"}, 
            status_code=500
        )

@app.post("/classify_all/")
async def classify_all(
    file: UploadFile = File(...), 
    channels: str = Query(None, description="Каналы через запятую")
):
    """Классификация изображения"""
    contents = await file.read()
    
    try:
        results = classifier.classify_all(contents, channels)
        return JSONResponse(results)
    except Exception as e:
        return JSONResponse(
            {"error": f"Ошибка классификации: {str(e)}"}, 
            status_code=500
        )

@app.get("/health")
async def health_check():
    """Проверка состояния"""
    return {
        "status": "ok",
        "model_loaded": classifier is not None and classifier.model is not None
    }

@app.get("/classes")
async def get_classes():
    """Информация о классах"""
    if classifier:
        return {
            "classes": classifier.class_names,
            "colors": classifier.class_colors
        }
    return {"error": "Классификатор не инициализирован"}
