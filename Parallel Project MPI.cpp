#include <iostream>
#include <opencv2/opencv.hpp>
#include <ctime>
#include <mpi.h>

#define PIXEL_RANGE 256 // Range of pixel intensities

using namespace std;

int* inputImage(int* w, int* h, const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath);

    if (image.empty()) {
        cerr << "Error: Unable to load image at " << imagePath << endl;
        return nullptr;
    }

    *w = image.cols;
    *h = image.rows;
    int* input = new int[(*w) * (*h)];

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(i, j);
            input[i * image.cols + j] = (pixel[2] + pixel[1] + pixel[0]) / 3;
        }
    }

    return input;
}

void createImage(int* image, int width, int height, int index) {
    cv::Mat outputImage(height, width, CV_8UC1);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int pixelValue = image[i * width + j];
            pixelValue = min(20, max(1, pixelValue)); // Map pixel value to range [1, 20]
            outputImage.at<uchar>(i, j) = static_cast<uchar>(pixelValue);
        }
    }

    std::string outputPath = "C:/Desktop/Programming/parallel programming project/code/parallelProject/Data/Output/outputRes" + std::to_string(index) + ".png";
    bool success = cv::imwrite(outputPath, outputImage);
    if (!success) {
        std::cerr << "Failed to save the image to: " << outputPath << std::endl;
    }
    cout << "Result image saved: " << outputPath << endl;
}

int main() {
    int imageWidth = 0, imageHeight = 0;
    string imagePaths[] = { "C:/Desktop/Programming/parallel programming project/code/parallelProject/Data/Input/test.png"
        , "C:/Desktop/Programming/parallel programming project/code/parallelProject/Data/Input/rose.png"
    , "C:/Desktop/Programming/parallel programming project/code/parallelProject/Data/Input/clahe_2.png"
    , "C:/Desktop/Programming/parallel programming project/code/parallelProject/Data/Input/girl.png" };


    int* imageData = inputImage(&imageWidth, &imageHeight, imagePaths[3]);

    if (imageData == nullptr) {
        cerr << "Failed to load image. Exiting..." << endl;
        return -1;
    }

    int start_s = clock();

    int rank, num_procs;
    int image_size = imageWidth * imageHeight;
    int* sendcounts, * startIdx, * local_chunk;
    int local_histogram[PIXEL_RANGE] = { 0 };
    int global_histogram[PIXEL_RANGE] = { 0 };

    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    sendcounts = new int[num_procs]; // specify the size of the chunk assigned to each process
    startIdx = new int[num_procs];  // Determine where each process's data chunk starts in the global array

    if (rank == 0) {
        // Step 1: Count frequency of each pixel intensity
        int chunk_size = image_size / num_procs;
        int remainder = image_size % num_procs;

        for (int i = 0; i < num_procs; i++) {
            sendcounts[i] = (i < num_procs - 1) ? chunk_size : chunk_size + remainder;
            startIdx[i] = (i == 0) ? 0 : startIdx[i - 1] + sendcounts[i - 1];
        }
    }

    MPI_Bcast(sendcounts, num_procs, MPI_INT, 0, MPI_COMM_WORLD); // used to broadcast data from one process to all other processes
    MPI_Bcast(startIdx, num_procs, MPI_INT, 0, MPI_COMM_WORLD);


    local_chunk = new int[sendcounts[rank]];


    MPI_Scatterv(imageData, sendcounts, startIdx, MPI_INT, local_chunk, sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    // Step 1: Compute local histogram
    for (int i = 0; i < sendcounts[rank]; i++) {
        local_histogram[local_chunk[i]]++;
    }

    MPI_Reduce(local_histogram, global_histogram, PIXEL_RANGE, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double global_probabilities[PIXEL_RANGE] = { 0.0 };
    if (rank == 0) {
        // Step 2: Compute probabilities
        for (int i = 0; i < PIXEL_RANGE; i++) {
            global_probabilities[i] = static_cast<double>(global_histogram[i]) / image_size;
        }

        // Step 3: Compute cumulative probabilities
        for (int i = 1; i < PIXEL_RANGE; i++) {
            global_probabilities[i] += global_probabilities[i - 1];
        }
    }

    // Broadcast cumulative probabilities to all processes
    MPI_Bcast(global_probabilities, PIXEL_RANGE, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Step 4: Map pixel intensities to range [1, 20]
    for (int i = 0; i < sendcounts[rank]; i++) {
        local_chunk[i] = static_cast<int>(global_probabilities[local_chunk[i]] * 20.0);
    }

    // Gather results
    MPI_Gatherv(local_chunk, sendcounts[rank], MPI_INT, imageData, sendcounts, startIdx, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        createImage(imageData, imageWidth, imageHeight, 2);
        delete[] imageData;
    }

    int stop_s = clock();
    if (rank == 0) {
        double totalTime = (stop_s - start_s) / double(CLOCKS_PER_SEC) * 1000;
        cout << "Parallel processing time: " << totalTime << " ms" << endl;
    }

    delete[] sendcounts;
    delete[] startIdx;
    delete[] local_chunk;

    MPI_Finalize();



    // Series Code //


    int imageW = 0, imageH = 0;

    int* imageD = inputImage(&imageW, &imageH, imagePaths[3]);

    int imageS = imageH * imageW;

    if (imageD == nullptr) {
        cerr << "Failed to load image. Exiting..." << endl;
        return -1;
    }

    int start = clock();

    int histogram[PIXEL_RANGE] = { 0 };

    for (int i = 0; i < imageS; i++) {
        histogram[imageD[i]]++;
    }

    double probabilities[PIXEL_RANGE] = { 0.0 };

    for (int i = 0; i < PIXEL_RANGE; i++) {
        probabilities[i] = static_cast<double>(histogram[i]) / imageS;
    }

    for (int i = 1; i < PIXEL_RANGE; i++) {
        probabilities[i] += probabilities[i - 1];
    }

    for (int i = 0; i < imageS; i++) {
        imageD[i] = static_cast<int>(probabilities[imageD[i]] * 20.0);
    }

    createImage(imageD, imageW, imageH, 1);

    int stop = clock();
    double totalTime = (stop - start) / double(CLOCKS_PER_SEC) * 1000;
    cout << "Sequential processing time: " << totalTime << " ms" << endl;

    delete[] imageD;


    return 0;
}
