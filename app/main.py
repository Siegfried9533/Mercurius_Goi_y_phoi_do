import json
import faiss
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import torch
from typing import List, Dict, Any, Optional
from PIL import Image
from pathlib import Path
import io

BASE_DIR = Path(__file__).resolve().parent.parent 

# --- (2) TẠO ĐƯỜNG DẪN TÀI SẢN (ASSETS) ---
METADATA_FILE = BASE_DIR / "assets" / "metadata_api.json"
FAISS_INDEX_FILE = BASE_DIR / "assets" / "faiss_index.index"
MODEL_NAME = 'clip-ViT-B-32-multilingual-v1'

#Ngưỡng đánh giá
ASSESSMENT_THRESHOLD = 0.85

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--- API đang khởi động trên thiết bị: {device} ---")

# Tạo ứng dụng FastAPI
app = FastAPI(
    title="API Gợi ý Trang phục",
    description="Một API sử dụng CLIP và FAISS để gợi ý trang phục dựa trên dáng người, sự kiện và thời tiết."
)

# Tạo một "global context" để giữ các model đã load
# Thay vì load lại mỗi khi có request, load 1 lần lúc API khởi động
model_cache = {}

class Measurements(BaseModel):
    """Mô hình dữ liệu cho số đo người dùng"""
    vong1: float = Field(..., description="Vòng 1 (cm)", example=90)
    vong2: float = Field(..., description="Vòng 2 (cm)", example=70)
    vong3: float = Field(..., description="Vòng 3 (cm)", example=100)
    # Thêm các số đo khác nếu cần (chiều cao, cân nặng...)

class RecommendationRequest(BaseModel):
    """Mô hình dữ liệu cho request gửi đến API"""
    measurements: Measurements
    event: str = Field(..., description="Sự kiện tham dự", example="Công sở")
    weather: str = Field(..., description="Mô tả thời tiết/mùa", example="mùa hè nóng")
    gender: str = Field(..., description="Giới tính người dùng", example="Nữ")

class OutfitResult(BaseModel):
    """Mô hình dữ liệu cho 1 kết quả trả về"""
    id: int
    image_path: str
    score: float # Độ tương đồng (càng cao càng tốt)

class RecommendationResponse(BaseModel):
    """Mô hình dữ liệu trả về cho người dùng"""
    prompt_used: str
    body_type_used: str
    total_results: int
    outfits: List[OutfitResult]

class SimilarLookInfo(BaseModel):
    image_path: Optional[str] = None
    event: Optional[List[str]] = None
    body_type: Optional[List[str]] = None

class AssessmentResponse(BaseModel):
    is_good_style: bool
    compatibility_percent: float
    message: str
    most_similar_look: SimilarLookInfo

#các hàm mở rộng
def encode_text(text: str) -> np.ndarray:
    """Vector hóa test và chuẩn hóa L2."""
    clip_model = model_cache['clip_model']

    query_vector = clip_model.encode([text], device=device, convert_to_tensor=True)
    query_vector_np = query_vector.cpu().numpy().astype('float32')

    # Chuẩn hóa L2
    faiss.normalize_L2(query_vector_np)
    return query_vector_np


def encode_image(image: Image.Image) -> np.ndarray:
    """Vector hóa hình ảnh và chuẩn hóa L2."""
    print("--- !!! ĐANG CHẠY HÀM ENCODE_IMAGE PHIÊN BẢN MỚI NHẤT (v5) !!! ---")
    clip_model = model_cache['clip_model']

    query_vector = clip_model.encode(image, device=device, convert_to_tensor=True)
    query_vector_np = query_vector.cpu().numpy().reshape(1, -1).astype('float32')

    # Chuẩn hóa L2
    faiss.normalize_L2(query_vector_np)
    return query_vector_np


def get_body_type_from_measurements(measurements: Measurements) -> str:
    """
    Hàm logic nghiệp vụ để chuyển đổi số đo thành dáng người.
    """
    # Đây CHỈ LÀ VÍ DỤ, logic này không chính xác
    if measurements.vong3 - measurements.vong1 > 10 and measurements.vong3 > measurements.vong2 + 25:
        return "Dáng quả lê"
    elif abs(measurements.vong1 - measurements.vong3) < 5 and measurements.vong2 < measurements.vong1 - 20:
        return "Dáng đồng hồ cát"
    elif measurements.vong2 > measurements.vong1:
        return "Dáng quả táo"
    else:
        return "Dáng chữ nhật"

#Khởi động API
@app.on_event("startup")
def load_models():
    """
    Hàm này sẽ tự động chạy 1 lần khi FastAPI khởi động.
    Load model CLIP, index FAISS và file metadata vào bộ nhớ (RAM).
    """
    print("Đang tải model CLIP...")
    model_cache['clip_model'] = SentenceTransformer(MODEL_NAME, device=device)
    print("Tải model CLIP thành công.")
    
    print("Đang tải FAISS index...")
    try:
        model_cache['faiss_index'] = faiss.read_index(str(FAISS_INDEX_FILE))
        print(f"Tải FAISS index thành công. Tổng số vector: {model_cache['faiss_index'].ntotal}")
    except Exception as e:
        print(f"LỖI NGHIÊM TRỌNG: Không thể load file {FAISS_INDEX_FILE}. Lỗi: {e}")
    
    print("Đang tải file Metadata...")
    try:
        with open(str(METADATA_FILE), 'r', encoding='utf-8') as f:
            model_cache['metadata'] = json.load(f)
        print(f"Tải metadata thành công. {len(model_cache['metadata'])} bản ghi.")
    except Exception as e:
        print(f"LỖI NGHIÊM TRỌNG: Không thể load file {METADATA_FILE}. Lỗi: {e}")
        
    print("--- KHỞI ĐỘNG API HOÀN TẤT ---")

#Tính năng gới ý
@app.post("/recommend/", response_model=RecommendationResponse)
def get_recommendations(request: RecommendationRequest, k_search: int = 100, k_final: int = 5):
    """
    Endpoint chính để nhận yêu cầu và trả về gợi ý trang phục.
    """
    # Lấy các model từ cache
    clip_model = model_cache.get('clip_model')
    faiss_index = model_cache.get('faiss_index')
    metadata = model_cache.get('metadata')

    if not all([clip_model, faiss_index, metadata]):
        raise HTTPException(status_code=503, detail="Máy chủ đang khởi động hoặc gặp lỗi, vui lòng thử lại sau giây lát.")

    # --- Bước A & B: Xử lý Input & Tạo Prompt ---
    target_body_type = get_body_type_from_measurements(request.measurements)
    target_gender = request.gender
    prompt_text = f"trang phục {request.event} cho {target_gender} mặc trong thời tiết {request.weather}"
    
    print(f"Nhận request: Prompt='{prompt_text}', BodyType='{target_body_type}'")

    # --- Bước C: Tìm kiếm Ngữ nghĩa (CLIP + FAISS) ---
    print(f"Đang vector hóa prompt...")
    query_vector_np = encode_text(prompt_text)

    print(f"Đang tìm kiếm {k_search} kết quả gần nhất...")
    # D: distances (khoảng cách/điểm tương đồng), I: indices (ID của FAISS)
    D, I = faiss_index.search(query_vector_np, k_search)
    
    search_indices = I[0] # Lấy danh sách ID (vì ta chỉ search 1 query)
    search_scores = D[0] # Lấy danh sách điểm tương đồng

    # --- Bước D: Lọc Cứng (Metadata) ---
    print("Đang lọc kết quả theo metadata...")
    final_results: List[OutfitResult] = []

    for i in range(len(search_indices)):
        faiss_id = search_indices[i]
        
        # Nếu faiss_id = -1 (có thể xảy ra), bỏ qua
        if faiss_id == -1:
            continue
            
        score = float(search_scores[i])

        try:
            # Lấy metadata tương ứng với ID
            item_metadata = metadata[faiss_id]
        except IndexError:
            print(f"CẢNH BÁO: Không tìm thấy metadata cho faiss_id {faiss_id}. Bỏ qua.")
            continue

        # --- Logic lọc ---
        
        # 1. Lọc giới tính
        if item_metadata.get('gender') != target_gender:
            continue # Bỏ qua nếu sai giới tính

        # 2. Lọc dáng người
        # Logic mới: 'target_body_type' phải nằm TRONG list 'body_type'
        if target_body_type not in item_metadata.get('body_type', []):
            continue # Bỏ qua nếu không hợp dáng người

        # 3. (Tùy chọn) Lọc nhiệt độ/event cứng nếu muốn
        # ...
        
        # Nếu vượt qua tất cả bộ lọc:
        final_results.append(
            OutfitResult(
                id=faiss_id,
                image_path=item_metadata.get('image_path'),
                score=score
            )
        )
        
        # Đã đủ số lượng kết quả cuối cùng -> dừng lọc
        if len(final_results) >= k_final:
            break

    print(f"Tìm kiếm hoàn tất. Trả về {len(final_results)} kết quả.")
    
    # --- Bước E: Trả về ---
    return RecommendationResponse(
        prompt_used=prompt_text,
        body_type_used=target_body_type,
        total_results=len(final_results),
        outfits=final_results
    )

# nhận xét hồ sơ
@app.post("/assess-style/", response_model=AssessmentResponse)
async def assess_style(file: UploadFile = File(..., description="Ảnh outfit người dùng cần đánh giá")):
    """
    Endpoint (Đánh giá): Nhận 1 ảnh, so sánh với CSDL "chuẩn mực" 
    và trả về % tương thích (đẹp/xấu).
    """
    # Lấy các model từ cache
    faiss_index = model_cache.get('faiss_index')
    metadata_list = model_cache.get('metadata')

    if not all([faiss_index, metadata_list]):
        raise HTTPException(status_code=503, detail="Máy chủ đang khởi động hoặc gặp lỗi, vui lòng thử lại sau giây lát.")
        
    print(f"Nhận request ĐÁNH GIÁ: {file.filename}")

    try:
        # Đọc ảnh người dùng
        contents = await file.read()
        user_image_raw = Image.open(io.BytesIO(contents))
        user_image = user_image_raw.convert("RGB")  # Đảm bảo ảnh ở định dạng RGB
        # Vector hóa ảnh (SỬA LỖI: đã bao gồm chuẩn hóa L2)
        query_vector_np = encode_image(user_image)
        
        # Tìm kiếm Image-to-Image (k=1: tìm 1 ảnh giống nhất)
        D, I = faiss_index.search(query_vector_np, k=1)
        
        similarity_score = float(D[0][0])
        best_match_id = int(I[0][0])
        
        # Tra cứu metadata của ảnh giống nhất
        best_match_info = metadata_list[best_match_id]
        
        # Đánh giá
        if similarity_score >= ASSESSMENT_THRESHOLD:
            is_good_style = True
            message = f"Đánh giá: Rất phù hợp (Tương thích {similarity_score*100:.1f}%)"
        else:
            is_good_style = False
            message = f"Kiến nghị: Chưa phù hợp (Tương thích {similarity_score*100:.1f}%)"
            
        return AssessmentResponse(
            is_good_style=is_good_style,
            compatibility_percent=round(similarity_score * 100, 2),
            message=message,
            most_similar_look=SimilarLookInfo(
                image_path=best_match_info.get("image_path"),
                event=best_match_info.get("event"),
                body_type=best_match_info.get("body_type")
            )
        )

    except Exception as e:
        print(f"LỖI xử lý ảnh: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý ảnh: {str(e)}")


# --- (Tùy chọn) Endpoint để kiểm tra API có "sống" không ---
@app.get("/health")
def health_check():
    return {"status": "ok"}

# --- Lệnh để chạy (dùng cho debug) ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)