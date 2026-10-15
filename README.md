# Seat Sync — Smart Library Seat Management System

**An AI-powered library seat monitoring and real-time availability platform using YOLO-based computer vision.**

Seat Sync helps students check library seat availability through a live-updating website. A mobile phone acts as the camera source, an object detection model identifies and classifies seats, and the student interface displays a privacy-preserving visual seat map.

The goal is to make finding an available library seat easier, faster, and more convenient.

## Project Overview

Libraries can be difficult to navigate when students do not know which seats are available. Manually checking seating areas wastes time and makes it difficult to understand current occupancy.

Seat Sync addresses this problem by combining computer vision, live video processing, backend services, and a web-based seating interface. The system analyses camera frames and updates seat availability so that students can make informed decisions before entering the library.

**Key principle:** Students see a mapped representation of seat availability, not the actual camera feed.

## Objectives

- Automate library seat occupancy monitoring using computer vision.
- Use a mobile phone as a live camera source for testing and monitoring.
- Detect seating regions and classify their occupancy status.
- Map detected seats to a consistent, easy-to-understand visual layout.
- Update seat availability dynamically on the student-facing website.
- Protect student privacy by displaying seat status rather than raw camera footage.
- Provide a modular foundation for future improvements in detection accuracy, performance, and deployment.

## Key Features

### AI-Based Seat Detection

Seat Sync uses YOLO-based object detection models to analyse images and video frames. The application includes YOLO26 and YOLOv8 model assets for its detection workflows.

The detection pipeline is designed to identify relevant objects and support seat discovery and occupancy monitoring.

### Mobile Camera Integration

A mobile phone supplies the live video stream. This allows the system to be tested without requiring a dedicated camera installation.

The camera is kept in a fixed position during initial testing to provide a consistent view of the seating area.

### Dynamic Seat Availability

The backend processes incoming frames and provides updated occupancy information to the website. The student interface uses these updates to display the latest available and occupied seats.

### Privacy-Preserving Student Interface

Students are shown a simplified seating map with visual seat indicators instead of the live camera footage. This separates the monitoring interface from the student-facing experience.

### Visual Seat Mapping

Detected seating regions are associated with positions in a mapped layout. The resulting interface represents seats as individual boxes, making it easier to identify their relative positions and availability.

### Multiple Detection Model Support

The project contains YOLO26 and YOLOv8 model assets. A configurable detection pipeline supports the use of primary and fallback model weights where enabled by the application configuration.

## System Architecture

The system follows a modular processing pipeline:

```text
Mobile Phone Camera
        |
        v
Live Video Stream
        |
        v
Backend Video Processing
        |
        v
YOLO-Based Object Detection
        |
        v
Seat Discovery and Occupancy Analysis
        |
        v
Seat Mapping and Status Updates
        |
        v
Student Website
        |
        v
Masked Seat Availability Layout
```

The administrator uses the monitoring interface to work with the camera feed and detection results. Students access the mapped seating layout to check availability without viewing the raw camera stream.

## Hardware Requirements

The following are the intended development and testing requirements. Actual performance depends on the device, model configuration, video resolution, and implementation.

| Component | Requirement |
|---|---|
| Development laptop | Windows laptop or compatible computer |
| Processor | Modern multi-core CPU |
| AI acceleration | Optional; depends on the supported runtime |
| Memory | 8 GB RAM recommended; 16 GB preferred |
| Camera source | Smartphone with a camera-streaming application |
| Network | Wi-Fi or mobile hotspot connecting the phone and laptop |
| Storage | Sufficient space for model weights, dependencies, and test data |
| Display | Browser capable of running the web dashboard |

The project can be tested without an NVIDIA GPU by using a supported CPU inference configuration. Available Intel graphics or AI acceleration should only be used when the installed runtime and model format support it.

## Software Requirements

| Software | Purpose |
|---|---|
| Python 3.11 | Backend development and model inference environment |
| Ultralytics | YOLO model loading and object detection |
| PyTorch | Model inference framework used by the YOLO pipeline |
| OpenCV | Image and video-frame processing |
| FastAPI | Backend API and application services |
| Uvicorn | ASGI server for running the backend |
| React and TypeScript | Frontend application and interactive dashboard |
| Node.js and npm | Frontend dependencies and development server |
| Git | Source control and project collaboration |
| Modern web browser | Accessing the administrator and student interfaces |

Use the dependency versions specified by the project configuration files wherever available.

## Technology Stack

- **Artificial Intelligence:** YOLO26, YOLOv8, Ultralytics, PyTorch
- **Computer Vision:** OpenCV and image/video processing
- **Backend:** Python, FastAPI, Uvicorn
- **Frontend:** React, TypeScript
- **Camera Input:** Smartphone-based video stream
- **Communication:** HTTP APIs and the configured live-update mechanism
- **Version Control:** Git and GitHub

## How It Works

1. **Camera setup:** A smartphone is positioned to capture the library seating area and connected to the same network as the computer running the application.
2. **Video acquisition:** The backend receives frames from the configured camera stream.
3. **Object detection:** The configured YOLO model analyses frames to detect relevant objects.
4. **Seat discovery:** The detection and mapping workflow identifies seating regions and associates them with positions in the mapped layout.
5. **Occupancy analysis:** The application uses its configured detection and occupancy logic to determine the status of mapped seats.
6. **Backend updates:** The latest seat-status information is made available to the website.
7. **Student display:** The student interface presents the mapped seat layout and current availability without exposing the raw camera feed.

The reliability of occupancy classification depends on camera placement, lighting, occlusion, model weights, and the occupancy logic used by the application.

## Installation and Setup

### Prerequisites

Install Python 3.11 and Node.js on your computer. Clone the repository and open a terminal in the project directory.

```bash
git clone https://github.com/Gineeth4/Seat-Sync-Colab.git
cd Seat-Sync-Colab
```

### Backend Setup

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Upgrade pip and install the project's declared Python dependencies.

```powershell
python -m pip install --upgrade pip
```

If the repository contains a `requirements.txt` file:

```powershell
python -m pip install -r requirements.txt
```

If the repository instead uses `pyproject.toml`, follow the installation instructions defined there.

Start the backend using the entry point and command specified by the project. For a FastAPI application whose entry point is `service/app.py` and whose application object is named `app`, the command would be:

```powershell
python -m uvicorn service.app:app --reload
```

Check the project's actual entry point if the command does not match its structure.

### Frontend Setup

Open a separate terminal in the project root and enter the frontend directory:

```powershell
cd dashboard
npm install
npm run dev
```

Open the local URL printed by the frontend development server.

If the project uses a different frontend directory or script, follow its `package.json`.

### Camera Setup

1. Connect the smartphone and development laptop to the same local network.
2. Start a compatible mobile camera-streaming application.
3. Obtain the stream URL provided by that application.
4. Configure the application with the correct URL using its supported settings.
5. Start the backend and open the administrator interface.
6. Verify that frames are received before testing detection and mapping.
7. Open the student interface separately and confirm that it displays the mapped seat layout rather than the raw video.

The exact stream URL and configuration method depend on the mobile camera application being used.

## Model Weights

The repository contains model assets for YOLO26 and YOLOv8. Their roles should be determined from the current application configuration.

Typical project paths include:

```text
weights/
├── yolo26n.pt
├── yolov8n.pt
└── seat_yolov8n.pt
```

The actual files may differ depending on which checkpoints are included in the repository.

**Important:** General-purpose pretrained weights and custom-trained seat-classification weights are not interchangeable. The custom checkpoint must have the class definitions and training appropriate for the application's occupancy logic.

Do not assume that a model detects empty and occupied seats correctly merely because it loads successfully. Validate the model on representative library images and live video.

## Testing and Evaluation

The following tests are recommended when validating the application:

| Test | Expected outcome |
|---|---|
| Backend startup | Application starts without import or configuration errors |
| Camera connectivity | Frames are received from the configured stream |
| Seat detection | Relevant seating regions are detected |
| Empty-seat scenario | Empty seats are represented as available when correctly classified |
| Occupied-seat scenario | Occupied seats are represented as occupied when correctly classified |
| Lighting variation | Detection behaviour is evaluated under bright and dim conditions |
| Student privacy | Raw camera footage is not displayed in the student interface |
| Live updates | Changes in occupancy are reflected in the mapped layout |
| Mapping consistency | Detected seating positions remain associated with the correct mapped seats |
| Network interruption | The application handles loss of camera connectivity without crashing |

For quantitative evaluation, use a separately labelled test dataset and record precision, recall, mAP, occupancy-classification accuracy, inference latency, and update delay. Do not report numerical performance claims without measurements.

## Current Scope and Limitations

- Detection quality depends on the model weights and the data used to train or validate them.
- A fixed camera position simplifies mapping and makes seat-position tracking more consistent.
- Significant camera movement may require remapping or recalibration.
- Occlusion, overlapping chairs, poor lighting, and changes in viewpoint can affect occupancy estimates.
- The student map is only as reliable as the seat mapping and occupancy-update pipeline.
- Model inference speed varies with hardware, input resolution, and runtime configuration.
- Production deployment would require additional testing for reliability, scalability, privacy, and operational monitoring.

## Future Enhancements

- Improve the training dataset with diverse library environments and lighting conditions.
- Add temporal smoothing to reduce rapid fluctuations in seat status.
- Improve automatic seat-map calibration and recovery after camera movement.
- Measure and optimise end-to-end latency on supported hardware.
- Add confidence-aware status handling for uncertain detections.
- Expand testing across different camera angles, seat designs, and occupancy patterns.

## Contributors

**1. Dhruv Vadher** — Project Leader and ML Developer

**2. Gineeth R** — Project Contributor

## Repository

GitHub: [Gineeth4/Seat-Sync-Colab](https://github.com/Gineeth4/Seat-Sync-Colab)

## Acknowledgements and References

- [Ultralytics Documentation](https://docs.ultralytics.com/) — YOLO models, inference, and training.
- [FastAPI Documentation](https://fastapi.tiangolo.com/) — Backend API development.
- [OpenCV Documentation](https://docs.opencv.org/) — Image and video processing.
- [React Documentation](https://react.dev/) — Frontend development.
- [PyTorch Documentation](https://pytorch.org/docs/) — Deep-learning framework.
- [GitHub Documentation](https://docs.github.com/) — Repository management and Markdown documentation.

---

*Seat Sync — Making library seat availability easier to see, understand, and access.*
