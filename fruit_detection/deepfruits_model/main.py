import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

def load_model():
    # Load DeepFruits model
    model = torch.load('deepfruits_model.pth')
    model.eval()
    
    # Load class names
    with open('deepfruits_classes.txt', 'r') as f:
        class_names = f.read().strip().split('\n')
    
    return model, class_names

def detect_fruits(frame, model, class_names):
    # Convert to PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Preprocess image
    input_tensor = transform(image).unsqueeze(0)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_tensor)
        _, predicted = torch.max(outputs, 1)
        confidence = torch.nn.functional.softmax(outputs, dim=1)[0][predicted]
    
    if confidence > 0.5:
        label = f"{class_names[predicted]}: {confidence:.2f}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return frame

def show_webcam():
    model, class_names = load_model()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame = detect_fruits(frame, model, class_names)
        cv2.imshow('DeepFruits Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    show_webcam() 