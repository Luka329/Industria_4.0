xfrom ultralytics import YOLO

# Carga tu modelo recién entrenado
model = YOLO("best.pt")

# Ejecuta la predicción con tu cámara web
results = model.predict(source="0", show=True, conf=0.5)
