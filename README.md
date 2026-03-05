# CUDA Based Handwritten Digit Analyzer

A high-performance **handwritten digit recognition system implemented using CUDA and C**.
This project trains and tests a **Convolutional Neural Network (CNN)** on the **MNIST dataset** while leveraging **GPU parallel computing** to accelerate computations.

The system demonstrates how **CUDA kernels can be used to implement neural network operations such as activation functions and matrix computations efficiently on NVIDIA GPUs.**

---

# Project Overview

Handwritten digit recognition is a fundamental problem in **machine learning and computer vision**. In this project, a neural network is implemented **from scratch using CUDA**, enabling GPU acceleration for faster training and inference.

The model is trained using the **MNIST dataset**, which contains thousands of labeled handwritten digit images (0–9).

This project demonstrates:

* GPU parallelization using **CUDA**
* Implementation of **neural network operations**
* Efficient processing of **large datasets**
* Performance benefits of **GPU acceleration**

---

# Features

* Handwritten digit classification using the **MNIST dataset**
* GPU accelerated computation using **CUDA**
* Neural network implementation from scratch in **C + CUDA**
* Training and testing functionality
* Demonstration video showing the working of the system
* Efficient GPU utilization for faster learning

---

# Dataset

This project uses the **MNIST handwritten digit dataset**, which contains:

* **60,000 training images**
* **10,000 testing images**
* Each image is **28 × 28 grayscale**

Dataset files included in the repository:

* `trainingSet.tar.gz`
* `testSet.tar.gz`
* `train-labels-idx1-ubyte.gz`

---

# Project Structure

```
CUDA-based-handwritten-digit-analyzer
│
├── src/                         # CUDA and C source files
│   ├── *.cu
│   ├── *.c
│   └── Makefile
│
├── trainingSet.tar.gz           # MNIST training dataset
├── testSet.tar.gz               # MNIST testing dataset
├── train-labels-idx1-ubyte.gz   # Label file
│
├── Video-CUDA Based Handwritten digit analyzer.mp4
├── README.md
```

---

# Prerequisites

To run this project you need:

* **NVIDIA GPU**
* **CUDA Toolkit installed**
* **C/C++ Compiler**
* **Make utility**

The project can run on **any operating system that supports CUDA**, such as:

* **Windows**
* **Linux**
* **WSL (Windows Subsystem for Linux)**

Example configuration used during development:

* NVIDIA GeForce GTX 1050 Ti (4GB GPU)
* NVIDIA CUDA Toolkit installed

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/CUDA-based-handwritten-digit-analyzer.git
cd CUDA-based-handwritten-digit-analyzer/src
```

Compile the project:

```bash
make
```

This will generate the executable:

```
mnist-cnn-gpu
```

---

# Running the Program

## Training the Model

Run the program with the number of training iterations:

```bash
./mnist-cnn-gpu [max_iter]
```

Example:

```bash
./mnist-cnn-gpu 100
```

Where `max_iter` represents the number of training iterations over the dataset.

---

## Testing the Model

Run the program **without arguments** to test the trained model.

```bash
./mnist-cnn-gpu
```

---

# Demo Video

A demonstration video showing the working of the **CUDA Based Handwritten Digit Analyzer** is included in this repository.

The video demonstrates:

* Model training process
* Digit classification
* Testing performance

---

# Training and Classification

During training, the system performs:

* Feature extraction from digit images
* Forward propagation through the neural network
* Weight updates using learning iterations

During classification, the trained model predicts the **digit label (0–9)** for unseen handwritten images.

---

# Technologies Used

* **CUDA**
* **C Programming**
* **GPU Parallel Computing**
* **Machine Learning Concepts**
* **MNIST Dataset**

---

# Applications

* Optical Character Recognition (OCR)
* Automated form processing
* Digit recognition systems
* AI and GPU computing research

---

# Future Improvements

Possible enhancements include:

* Improving CNN architecture for higher accuracy
* Supporting real-time handwritten input
* Adding a graphical interface
* Extending the model for character recognition

---

# Author

**Khushi Gupta**
 
