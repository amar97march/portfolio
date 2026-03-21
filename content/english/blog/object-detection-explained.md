---
title: "Object Detection: Drawing Boxes Around Everything"
date: 2026-12-29T10:00:00+05:30
draft: false
description: "Object detection goes beyond classification — it finds every object in an image and draws a bounding box around it. This post explains the architectures, metrics, and practical applications of modern object detection."
tags: ["Computer Vision", "Object Detection", "YOLO", "Deep Learning", "Neural Networks"]
categories: ["Computer Vision"]
image: "https://picsum.photos/seed/object-detection-explained-cover/1200/630"
keywords: ["object detection", "YOLO", "SSD", "Faster R-CNN", "bounding box", "anchor boxes", "mAP", "real-time detection"]
---

Image classification answers a simple question: "What is in this image?" But the real world rarely presents us with images containing a single, neatly centered object. A photograph of a busy street might contain cars, pedestrians, traffic lights, bicycles, and a stray dog — all in a single frame.

**Object detection** answers a much harder question: "What objects are in this image, and where exactly are they?"

Instead of producing a single label, an object detection model outputs a list of objects, each with a **class label** and a **bounding box** — a rectangle that specifies the object's location and size within the image.

This is the technology that powers autonomous driving, security surveillance, medical imaging, and augmented reality. And understanding how it works is essential for anyone serious about Computer Vision.

---

### From Classification to Detection

Let us build an intuition for why detection is harder than classification.

In classification, the model sees an image and outputs one label. The entire image is the input, and the entire image contributes to the output. The model does not need to know *where* things are.

In detection, the model must:

1. **Locate** every object of interest in the image.
2. **Classify** each located object.
3. **Handle** an arbitrary number of objects per image.
4. **Suppress** duplicate detections of the same object.

This turns what was a simple "image in, label out" problem into a far more complex "image in, list of (box, label, confidence) out" problem.

---

### The Two-Stage Approach: R-CNN Family

The first successful deep learning approach to object detection was the **R-CNN** (Regions with CNN features) family, developed by Ross Girshick and colleagues starting in 2014.

**R-CNN (2014).** The original approach used a classical algorithm called Selective Search to propose about 2,000 candidate regions in the image. Each region was then resized and fed through a CNN for classification. It was accurate but painfully slow — about 47 seconds per image.

**Fast R-CNN (2015).** Instead of running the CNN on each region separately, Fast R-CNN runs the CNN once on the entire image to produce a shared feature map. Regions of interest are then projected onto this feature map and classified. This was about 10x faster.

**Faster R-CNN (2015).** Replaced Selective Search with a **Region Proposal Network (RPN)** — a small neural network that proposes regions directly from the feature map. This made the entire pipeline end-to-end trainable and brought inference time down to about 5 frames per second.

The two-stage approach (propose regions, then classify them) is accurate but inherently limited in speed because of the sequential nature of the process.


![Computer vision analyzing visual data](https://picsum.photos/seed/object-detection-explained-1/800/450)

---

### The One-Stage Revolution: YOLO and SSD

The real game-changer came with **one-stage detectors** — models that predict bounding boxes and class labels in a single forward pass through the network.

**YOLO — You Only Look Once (2016).** Joseph Redmon's YOLO was revolutionary in its simplicity. Instead of proposing regions first, YOLO divides the image into a grid (e.g., 7x7) and has each grid cell predict a fixed number of bounding boxes along with their class probabilities. The entire detection happens in one shot.

YOLO was fast — 45 frames per second on a GPU, fast enough for real-time video. The tradeoff was accuracy, especially for small objects and objects that overlap.

The YOLO architecture has evolved significantly through multiple versions:

- **YOLOv2 (2017):** Added batch normalization, anchor boxes, and multi-scale training.
- **YOLOv3 (2018):** Used a deeper backbone (Darknet-53) and made predictions at three different scales.
- **YOLOv5 (2020):** PyTorch-based, highly optimized for practical use.
- **YOLOv8 (2023):** The latest from Ultralytics, with improved accuracy and a cleaner API.

**SSD — Single Shot MultiBox Detector (2016).** Similar in spirit to YOLO, SSD makes predictions at multiple scales from different layers of the network. This helps it detect both large and small objects effectively.

Here is how you can run YOLOv8 inference with just a few lines of code:

```python
from ultralytics import YOLO

# Load a pretrained YOLOv8 model
model = YOLO("yolov8n.pt")  # nano model for speed

# Run inference on an image
results = model("street_scene.jpg")

# Print detections
for result in results:
    for box in result.boxes:
        cls = result.names[int(box.cls)]
        conf = float(box.conf)
        coords = box.xyxy[0].tolist()
        print(f"{cls}: {conf:.2f} at {coords}")

# Save annotated image
results[0].save("output.jpg")
```

In just a few lines, you have a working object detection system that can find and label objects in any image.

---

### Key Concepts in Object Detection

**Anchor Boxes.** Most modern detectors use predefined anchor boxes — templates of different sizes and aspect ratios that the model uses as starting points for predicting bounding boxes. The model learns to adjust these anchors to fit the actual objects.

**Intersection over Union (IoU).** The standard metric for measuring how well a predicted box matches the ground truth box. IoU = (area of overlap) / (area of union). An IoU of 0.5 or higher is typically considered a "hit."

**Non-Maximum Suppression (NMS).** A post-processing step that removes duplicate detections. When multiple boxes predict the same object, NMS keeps the one with the highest confidence and removes the rest.

**Mean Average Precision (mAP).** The standard evaluation metric for object detection. It computes the average precision for each class across different IoU thresholds and averages them. mAP@0.5 uses a 0.5 IoU threshold. mAP@0.5:0.95 averages across thresholds from 0.5 to 0.95 in steps of 0.05 — this is the tougher benchmark.


![Image processing and object recognition](https://picsum.photos/seed/object-detection-explained-2/800/450)

---

### Modern Architectures

Beyond YOLO and SSD, several important architectures have emerged:

**Feature Pyramid Networks (FPN).** A general-purpose architecture for building multi-scale feature maps. It is used as a building block in many modern detectors.

**DETR (2020).** Facebook AI's Detection Transformer applies the Transformer architecture to object detection. It eliminates the need for anchor boxes and NMS by using a set-based loss function and learned object queries. It is elegant but was initially slower to train than CNN-based detectors.

**RT-DETR (2023).** A real-time version of DETR that matches or exceeds YOLO in speed while maintaining the architectural elegance of Transformers.

---

### Practical Applications

Object detection is everywhere:

- **Autonomous driving.** Detecting pedestrians, vehicles, traffic signs, and lane markings in real-time is critical for safe navigation.
- **Surveillance.** Detecting people, weapons, or suspicious behavior in security camera feeds.
- **Retail.** Tracking products on shelves, detecting stock-outs, and powering cashier-less checkout.
- **Medical imaging.** Detecting tumors, lesions, and abnormalities in X-rays, CT scans, and MRIs.
- **Sports analytics.** Tracking players and the ball in real-time for performance analysis.
- **Wildlife monitoring.** Detecting and counting animals in drone or camera trap footage.


![Visual AI technology detecting patterns in images](https://picsum.photos/seed/object-detection-explained-3/800/450)

---

### The Speed-Accuracy Tradeoff

Object detection always involves a tradeoff between speed and accuracy. Two-stage detectors like Faster R-CNN are more accurate but slower. One-stage detectors like YOLO are faster but may miss small or overlapping objects.

The choice depends on the application. For a self-driving car, you need both speed and accuracy — missing a pedestrian is not an option. For offline medical image analysis, speed matters less but accuracy is critical.

Modern architectures like YOLOv8 and RT-DETR are closing this gap, offering near-real-time inference with accuracy that approaches two-stage detectors.

---

### What Comes Next

Object detection tells us *what* is in an image and *where* it is, but only at the level of bounding boxes — rough rectangles. What if we need to know the exact shape of each object, down to the pixel?

That is **image segmentation**, and it is the topic of our next post. We will go from drawing boxes to drawing exact outlines.

— Amar Singh
