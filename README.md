# ENGLISH

# Image Processor AI (Logo & Background Remover)
====================================================

A Python-based application for image processing using AI LaMa Inpainting to remove 
objects or logos, along with watermark and background removal features. This 
application is specifically optimized for MacBook Apple Silicon (M1/M2) devices 
with custom CPU/MPS stability patches[cite: 1].

## Project Description
This project was developed to facilitate the automatic cleaning of image assets 
without the need for heavy editing software. It utilizes the TorchScript Large 
Mask Inpainting (LaMa) model for natural and seamless background reconstruction.

## Key Features
*   AI Magic Eraser: Remove objects or logos using an interactive brush tool[cite: 1].
*   Watermark Remover: Threshold-based algorithm specifically designed for white 
    or grey semi-transparent watermarks.
*   Background Remover: Supports both the 'rembg' (AI-powered) library and 
    OpenCV GrabCut for transparency[cite: 1].
*   M2 Optimization: Automatically bypasses CUDA/MPS issues to ensure the 
    application does not crash during the inpainting process[cite: 1].

## System Requirements & Installation
This application requires Python 3.11 or 3.12 for optimal GUI stability.

1.  Create a Virtual Environment:
    python3 -m venv .venv
    source .venv/bin/activate

2.  Install Core Dependencies:
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    pip install numpy Pillow opencv-python rembg

## How to Run
1.  Execute the main script:
    python3 Proses_Gambar_1.2.py

2.  Automatic Model Download: 
    Upon the first run, the system will automatically download the 'big-lama.pt' 
    model (~200MB) to the ~/.cache/simple-lama-inpainting/ directory[cite: 1].

3.  Usage Workflow:
    - Click 'OPEN IMAGE' (BUKA GAMBAR) to upload a photo.
    - Select the 'AI MAGIC ERASER' tab.
    - Click 'OPEN BRUSH TOOL' (BUKA KUAS PENGHAPUS), highlight the area to be 
      removed, then click 'Generate AI'.
    - Click 'SAVE RESULT' (SIMPAN HASIL) once the process is complete.

##  echnical Notes (Mac M1/M2)
Due to limitations of the MPS (Metal Performance Shaders) operator in certain 
TorchScript models, this application is locked to run on the CPU by default[cite: 1]. 
This is implemented to prevent 'TorchScript interpreter failure' errors when 
processing images with specific resolutions[cite: 1].

## Author
*   Vincent Richard Tanuhariono
*   Fullstack Developer & Junior Data Analyst[cite: 1]



# INDONESIA

# Image Processor AI (Logo & Background Remover)
====================================================

Aplikasi berbasis Python untuk pengolahan gambar menggunakan AI LaMa Inpainting 
untuk menghapus objek atau logo, serta fitur penghapusan watermark dan background. 
Aplikasi ini telah dioptimalkan khusus untuk perangkat MacBook Apple Silicon (M1/M2) 
dengan patch stabilitas CPU/MPS[cite: 1].

## Deskripsi Proyek
Proyek ini dikembangkan untuk memudahkan pembersihan aset gambar secara otomatis 
tanpa memerlukan software editing yang berat. Menggunakan model TorchScript 
Large Mask Inpainting (LaMa) untuk rekonstruksi background yang natural.

## Fitur Utama
*   AI Magic Eraser: Menghapus objek/logo dengan kuas interaktif.
*   Hapus Watermark: Algoritma berbasis threshold untuk watermark putih/abu-abu.
*   Hapus Background: Mendukung library 'rembg' (AI) atau OpenCV GrabCut.
*   M2 Optimization: Bypass otomatis masalah CUDA/MPS untuk menjamin aplikasi 
    tidak crash saat inpainting.

## Persyaratan Sistem & Instalasi
Aplikasi ini membutuhkan Python 3.11 atau 3.12 untuk kestabilan GUI.

1. Buat Virtual Environment:
   python3 -m venv .venv
   source .venv/bin/activate

2. Instal Dependensi Utama:
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   pip install numpy Pillow opencv-python rembg

## Cara Menjalankan
1. Jalankan script utama:
   python3 Proses_Gambar_1.2.py

2. Unduh Model Otomatis: 
   Saat pertama kali dijalankan, sistem akan mengunduh model 'big-lama.pt' (~200MB) 
   ke folder ~/.cache/simple-lama-inpainting/.

3. Alur Penggunaan:
   - Klik 'BUKA GAMBAR' untuk mengunggah foto.
   - Pilih tab 'AI MAGIC ERASER'.
   - Klik 'BUKA KUAS PENGHAPUS', coret area logo, lalu klik 'Generate AI'.
   - Klik 'SIMPAN HASIL' setelah proses selesai.

## Catatan Teknis (Mac M1/M2)
Karena keterbatasan operator MPS pada model TorchScript tertentu, aplikasi ini 
secara default dikunci untuk berjalan pada CPU. Hal ini dilakukan untuk 
mencegah error 'TorchScript interpreter failure' saat memproses gambar dengan 
resolusi spesifik.

## Author
- Vincent Richard Tanuhariono
- Developer & Data Analyst
