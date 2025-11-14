import json
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
from tqdm import tqdm
import torch
import os

# --- 1. CẤU HÌNH---
LABEL_STUDIO_EXPORT_FILE = './data/label_studio_exports/metadata.json' 

BASE_IMAGE_FOLDER = "./data/lookbooks/female" 
MODEL_NAME = 'clip-ViT-B-32-multilingual-v1'

OUTPUT_VECTOR_FILE = './assets/image_vectors.npy'    # Cho FAISS
OUTPUT_METADATA_FILE = './assets/metadata_api.json'  # Cho API

def parse_label(label_data):
    if isinstance(label_data, dict):
        return label_data.get('choices', [])
    elif isinstance(label_data, str):
        return [label_data]
    elif isinstance(label_data, list):
        return label_data
    else:
        return []

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--- Đang sử dụng thiết bị: {device} ---")

print(f"Đang tải model CLIP: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME, device=device)
print("Tải model thành công.")

# --- Đọc file JSON thô  ---
print(f"Đang đọc file export của Label Studio từ {LABEL_STUDIO_EXPORT_FILE}...")
try:
    with open(LABEL_STUDIO_EXPORT_FILE, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)
except FileNotFoundError:
    print(f"LỖI: Không tìm thấy file {LABEL_STUDIO_EXPORT_FILE}.")
    exit()

print(f"Đã đọc thành công {len(raw_data)} bản ghi (task).")


all_vectors = []
final_metadata_list = []
faiss_id = 0

print("Bắt đầu quá trình phân tích metadata và vector hóa ảnh...")

for task in tqdm(raw_data, desc="Processing Tasks"):
    try:
        #  Lấy đường dẫn Label Studio
        ls_image_path = task.get('image')
        if not ls_image_path:
            print(f"\nCẢNH BÁO: Bỏ qua task ID {task.get('id')} vì 'image' rỗng.")
            continue

        # Lấy tên file có hash"
        file_name_with_hash = os.path.basename(ls_image_path)
        
        # Cắt bỏ 9 ký tự đầu (hash 8 ký tự + dấu gạch ngang)
        original_file_name = file_name_with_hash[9:] 

        # Tạo đường dẫn thực tế
        full_image_path = os.path.join(BASE_IMAGE_FOLDER, original_file_name)

        vector_list = model.encode([full_image_path], device=device)
        vector = vector_list[0]

    except FileNotFoundError:
        print(f"\nCẢNH BÁO: KHÔNG TÌM THẤY ẢNH:")
        continue 
    except Exception as e:
        print(f"\nLỖI khi xử lý ảnh {full_image_path}: {e}. Bỏ qua task ID {task.get('id')}.")
        continue

    # --- Phân tích nhãn (metadata) ---
    gender_label = task.get('gender')
    body_type_list = parse_label(task.get('body_type'))
    event_list = parse_label(task.get('event'))
    temp_list = parse_label(task.get('temperature'))

    metadata_record = {
        "id_faiss": faiss_id,
        "image_path": full_image_path, # Lưu đường dẫn đầy đủ, chính xác
        "gender": gender_label,
        "body_type": body_type_list, 
        "event": event_list,         
        "temperature": temp_list     
    }
    
    final_metadata_list.append(metadata_record)
    all_vectors.append(vector)
    
    faiss_id += 1

if not all_vectors:
    print("LỖI: Không có vector nào được tạo. Dừng chương trình.")
    exit()

vectors_np = np.array(all_vectors).astype('float32')
np.save(OUTPUT_VECTOR_FILE, vectors_np)
print(f"\n--- THÀNH CÔNG (1/2) ---")
print(f"Đã lưu {len(all_vectors)} vector vào file: {OUTPUT_VECTOR_FILE}")

with open(OUTPUT_METADATA_FILE, 'w', encoding='utf-8') as f:
    json.dump(final_metadata_list, f, ensure_ascii=False, indent=2)
print(f"--- THÀNH CÔNG (2/2) ---")
print(f"Đã lưu metadata đơn giản cho API vào file: {OUTPUT_METADATA_FILE}")
