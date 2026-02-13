from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import keras


app = FastAPI()

Model=tf.keras.models.load_model("saved_models\potato_modell.keras")

class_names=['Early Blight','Late Blight','Healthy']

@app.get("/ping")
async def ping():
    return {"message": "hello, I'm alive"}

def read_file_as_image(data) -> np.ndarray:
    try:
        image = np.array(Image.open(BytesIO(data)))
        return image
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    Im=await file.read()
    image = read_file_as_image(Im)  # Read file asynchronously     
    img=np.expand_dims(image,0)
    predictions=Model.predict(img)
    pred_class=class_names[np.argmax(predictions[0])]
    confidence=np.max(predictions[0])
    return{
        'class':pred_class,
        'confidence':float(confidence)
    }
    
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
