import cv2
import numpy as np

def load_model():
    with open('coco.names', 'r') as f:
        class_names = f.read().strip().split('\n')
    
    net = cv2.dnn.readNet(
        'yolov3.weights',
        'yolov3.cfg'
    )
    
    return net, class_names

def detect_objects(frame, net, class_names):
    height, width = frame.shape[:2]
    
    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    
    # Get output layer names
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Forward pass through the network
    outputs = net.forward(output_layers)
    
    # Initialize lists for detected objects
    boxes = []
    confidences = []
    class_ids = []
    
    # Process each detection
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            if confidence > 0.5:  # Confidence threshold
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                
                # Rectangle coordinates
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    # Apply non-maximum suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # Draw boxes
    for i in indices:
        i = i
        box = boxes[i]
        x, y, w, h = box
        
        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        label = f"{class_names[class_ids[i]]}: {confidences[i]:.2f}"
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame

def show_webcam():
    # Load the model
    net, class_names = load_model()
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect objects in the frame
        processed_frame = detect_objects(frame, net, class_names)
        
        # Display the processed frame
        cv2.imshow('Object Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start the webcam
show_webcam()


