import faiss
import numpy as np
import os

# --- Cấu hình ---
INPUT_VECTOR_FILE = 'assets/image_vectors.npy'
OUTPUT_INDEX_FILE = 'assets/faiss_index.index'

print(f"Đang đọc file vector từ: {INPUT_VECTOR_FILE}...")

try:
    vectors = np.load(INPUT_VECTOR_FILE)
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file {INPUT_VECTOR_FILE}.")
    print("Hãy đảm bảo bạn đã chạy script 'run_vectorization_final.py' thành công.")
    exit()

# 1. Đảm bảo vector là float32
vectors = vectors.astype('float32')

# 2. Chuẩn hóa L2 (Rất quan trọng cho IndexFlatIP)
print("Đang chuẩn hóa L2 (L2 Normalization) cho các vector...")
faiss.normalize_L2(vectors)

# Nó phải là 512 cho model 'clip-ViT-B-32-multilingual-v1'
d = vectors.shape[1] 
print(f"Đã load {vectors.shape[0]} vector, mỗi vector có {d} chiều.")

index = faiss.IndexFlatIP(d)

print("Đang thêm vector vào FAISS Index (việc này có thể mất chút thời gian)...")
index.add(vectors)

print(f"Thêm thành công. Tổng số vector trong index: {index.ntotal}")

print(f"Đang lưu index vào file: {OUTPUT_INDEX_FILE}...")
faiss.write_index(index, OUTPUT_INDEX_FILE)

print("--- HOÀN TẤT! ---")
print(f"File index đã được lưu tại: {OUTPUT_INDEX_FILE}")