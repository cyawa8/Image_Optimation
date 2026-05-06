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
