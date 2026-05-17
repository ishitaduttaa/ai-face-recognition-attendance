# AI-face-recognition-attendance
AI-powered attendance system using real-time face recognition, dual-camera entry/exit validation, and duration-based verification to prevent proxy attendance and automate classroom attendance management.

## Automated Student Attendance System Using Real-Time Face Recognition

A smart attendance management system that uses **real-time face recognition**, **dual-camera validation**, and **duration-based verification** to automate classroom attendance while preventing proxy attendance and false positives. 

## Features
* Real-time face recognition using **dlib** and **OpenCV**
* Dual-camera setup for **Entry & Exit Validation**
* Prevents proxy attendance
* Uses **128-dimensional facial embeddings**
* Euclidean distance threshold (`0.5`) for reliable recognition
* Minimum stay duration verification (`40 minutes`)
* Automatic CSV attendance generation
* Handles duplicate detections efficiently
* Low-cost hardware setup using standard webcams

## Tech Stack
* **Python**
* **OpenCV**
* **dlib**
* **face_recognition**
* **NumPy**
* **Pandas**

## System Workflow
1. Student videos are collected
2. Frames are extracted from videos
3. Faces are detected and cropped
4. Images are preprocessed and resized
5. Face embeddings are generated
6. Entry camera logs entry time
7. Exit camera logs exit time
8. Duration is calculated
9. Attendance CSV is generated automatically

## Working Principle

The system generates **128D facial embeddings** using a pre-trained CNN model from the `dlib` library. During recognition, Euclidean distance is calculated between live face embeddings and stored embeddings.

A student is marked **Present** only if:

* Face is detected at **Entry Camera**
* Face is detected at **Exit Camera**
* Euclidean distance < `0.5`
* Stay duration ≥ `40 minutes`

Otherwise, the student is marked **Absent**.

---

## Installation

```bash
git clone <your-repo-link>
cd attendance-system
```

Install dependencies:

```bash
pip install opencv-python
pip install face_recognition
pip install dlib
pip install numpy
pip install pandas
```

---

## Run the Project

### Step 1: Extract Frames

```bash
python v2p_10_.py
```

### Step 2: Detect & Crop Faces

```bash
python photo2faces.py
```

### Step 3: Resize Images

```bash
python p2_224x224_.py
```

### Step 4: Convert Images to RGB

```bash
python convert2rgb.py
```

### Step 5: Validate Dataset

```bash
python validate_dataset.py
```

### Step 6: Generate Face Encodings

```bash
python recog_face.py
```

### Step 7: Start Entry Recognition

```bash
python F_Recog.py
```

### Step 8: Start Exit Recognition

```bash
python F_Recog_exit.py
```

---

## Output

The system automatically generates:
* Entry attendance CSV
* Final attendance CSV containing:

  * Student Name
  * Date
  * Entry Time
  * Exit Time
  * Duration
  * Attendance Status

---

## Advantages
* Eliminates manual attendance
* Reduces classroom time wastage
* Prevents fake/proxy attendance

---

## Future Enhancements
* Web dashboard for attendance analytics
* ERP integration
* Liveness detection
---

## Authors
* Ishita Dutta
* Palash Dandapat
* Mangal Adikary
* Mahek Saha
* 
Department of Computer Science and Engineering
KIIT, Bhubaneswar

