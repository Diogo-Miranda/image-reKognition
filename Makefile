# Directory structure
COCO_DIR = fruit_detection/coco_model

# URLs for files
WEIGHTS_URL = https://pjreddie.com/media/files/yolov3.weights
CFG_URL = https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg
NAMES_URL = https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names

# Target files
WEIGHTS = $(COCO_DIR)/yolov3.weights
CFG = $(COCO_DIR)/yolov3.cfg
NAMES = $(COCO_DIR)/coco.names

.PHONY: all clean download

# Default target
all: download

# Create directory if it doesn't exist
$(COCO_DIR):
	mkdir -p $(COCO_DIR)

# Download files
download: $(COCO_DIR) $(WEIGHTS) $(CFG) $(NAMES)
	@echo "All files downloaded successfully!"

$(WEIGHTS):
	@echo "Downloading YOLOv3 weights..."
	wget -O $(WEIGHTS) $(WEIGHTS_URL)

$(CFG):
	@echo "Downloading YOLOv3 config..."
	wget -O $(CFG) $(CFG_URL)

$(NAMES):
	@echo "Downloading COCO names..."
	wget -O $(NAMES) $(NAMES_URL)

# Clean up downloaded files
clean:
	rm -f $(WEIGHTS) $(CFG) $(NAMES)

# Clean everything including directory
clean-all: clean
	rm -rf $(COCO_DIR) 