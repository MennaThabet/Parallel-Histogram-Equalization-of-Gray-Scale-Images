# Parallel Histogram Equalization for Grayscale Images

> ⚡ Image contrast enhancement using MPI and OpenCV

## 📌 Overview

Histogram equalization is a technique that improves the contrast in grayscale images by spreading out the most frequent intensity values.

This project implements both **sequential** and **parallel** versions of histogram equalization in **C++**. The parallel version uses **MPI** (Message Passing Interface) to distribute processing across multiple processors.

## 🧠 What It Does

- Reads grayscale images and converts color images to grayscale
- Calculates the frequency of each pixel intensity (0–255)
- Normalizes intensity values to the range `[1, 20]` using cumulative probability distribution
- Enhances image contrast
- Outputs and saves the resulting image

## 🚀 Technologies Used

- **C++**
- **MPI** (Parallel version)
- **OpenCV** (Image I/O and grayscale conversion)
- `MPI_Scatterv`, `MPI_Reduce`, `MPI_Gatherv` for distributed computing

## 🧪 Comparison of Versions

- ✅ **Sequential Version**:
  - Single process
  - Straightforward histogram equalization
  - Processes entire image in one pass

- ⚡ **Parallel Version**:
  - Divides image into chunks across MPI processes
  - Computes local histograms and merges them into a global histogram
  - Speeds up processing of large images

## 🖼️ Sample Output

```bash
Sequential processing time: 250 ms  
Parallel processing time: 90 ms  
Result image saved: outputRes1.png (sequential)  
Result image saved: outputRes2.png (parallel)

## 📂 File Structure

├── main.cpp                # Main code (sequential + parallel)
├── Input/                  # Original grayscale test images
├── Output/                 # Saved results
└── ...

## 🧮 Key Functions
- inputImage(): Loads and converts input image to grayscale
- createImage(): Saves processed result image
- MPI_Scatterv(): Distributes image chunks
- MPI_Reduce(): Aggregates local histograms
- MPI_Gatherv(): Reconstructs final image

## ⚙️ How to Run
1. Compile with OpenCV and MPI:
mpic++ main.cpp -o equalize `pkg-config --cflags --libs opencv4`

2. Run on multiple processes:
mpirun -np 4 ./equalize

3. Check Output/ folder for resulting enhanced images.
Modify paths if needed in imagePaths[] for your environment.

## 📌 Learning Outcomes
- Gained experience in parallelizing real-world algorithms using MPI
- Practiced low-level optimization and performance timing
- Applied OpenCV for visual validation and image processing
