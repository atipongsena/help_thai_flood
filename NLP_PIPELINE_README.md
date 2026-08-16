# Thai Flood Relief NLP Pipeline Documentation

เอกสารนี้อธิบายขั้นตอนการติดตั้งและใช้งาน NLP Pipeline สำหรับระบบช่วยเหลือผู้ประสบภัยน้ำท่วม

## 1. Prerequisites (สิ่งที่ต้องมี)

- **Python 3.9+** (แนะนำ 3.10 หรือ 3.11)
- **CUDA-capable GPU** (แนะนำสำหรับ Training BERT model, ถ้าไม่มีจะช้ามาก)
- **Git**

## 2. Setup & Installation (การติดตั้ง)

1.  **Clone Repository** (ถ้ายังไม่ได้ทำ)

    ```bash
    git clone <repository_url>
    cd help_thai_flood
    ```

2.  **Create Virtual Environment** (แนะนำ)

    ```bash
    # Windows
    python -m venv .venv
    .venv\Scripts\activate

    # Linux/Mac
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

    _หมายเหตุ: หากต้องการใช้ GPU กับ PyTorch ให้ตรวจสอบเวอร์ชันที่เหมาะสมกับ CUDA ของคุณที่ [pytorch.org](https://pytorch.org/get-started/locally/)_

4.  **Environment Variables**
    - สร้างไฟล์ `.env` จาก `.env.example`
    - ตั้งค่าตัวแปรต่างๆ (ถ้ามี) เช่น `FB_ACCESS_TOKEN` สำหรับการดึงข้อมูลจริง

## 3. Running the Pipeline (การรัน Pipeline)

สคริปต์หลักสำหรับรันทุกขั้นตอนคือ `run_pipeline.py`

### คำสั่งพื้นฐาน

| คำสั่ง                          | รายละเอียด                                                      |
| :------------------------------ | :-------------------------------------------------------------- |
| `python run_pipeline.py --all`  | รันทุกขั้นตอน (Prepare -> Train -> Evaluate) โดยใช้ Sample Data |
| `python run_pipeline.py --api`  | รัน API Server สำหรับใช้งานจริง                                 |
| `python run_pipeline.py --help` | ดูคำสั่งทั้งหมด                                                 |

### ขั้นตอนย่อย (Step-by-Step)

คุณสามารถรันแยกแต่ละขั้นตอนได้ดังนี้:

#### Step 1: Prepare Data (เตรียมข้อมูล)

สร้างข้อมูลตัวอย่างหรือแปลงข้อมูลจาก Database เป็น CSV สำหรับ Training

```bash
# ใช้ Sample Data (สำหรับทดสอบระบบ)
python run_pipeline.py --prepare

# ใช้ Real Data (ข้อมูลที่ Scrape มาใน database)
python run_pipeline.py --prepare --use-real-data
```

#### Step 2: Train Classical Models (เทรนโมเดลพื้นฐาน)

เทรนโมเดล Machine Learning พื้นฐาน (TF-IDF + SVM/Naive Bayes)

```bash
# เทรนเฉพาะโมเดลที่ดีที่สุด (TF-IDF + SVM)
python run_pipeline.py --train-classical

# รันการทดลองทั้งหมด 24 แบบ
python run_pipeline.py --train-classical --all-experiments
```

#### Step 3: Train BERT Model (เทรนโมเดล BERT)

Fine-tune WangchanBERTa model (ต้องการ GPU)

```bash
python run_pipeline.py --train-bert
```

#### Step 4: Evaluate (ประเมินผล)

แสดงผลลัพธ์การวัดผล (F1 Score, Accuracy) ของโมเดลทั้งหมด

```bash
python run_pipeline.py --evaluate
```

#### Step 5: Start API (เริ่มระบบ API)

รัน FastAPI server เพื่อให้บริการ Model

```bash
python run_pipeline.py --api
```

- API จะรันที่ `http://localhost:8000`
- Documentation: `http://localhost:8000/docs`

## 4. Directory Structure (โครงสร้างไฟล์)

- `run_pipeline.py`: สคริปต์หลักสำหรับควบคุม Pipeline
- `requirements.txt`: รายชื่อ Library ที่ต้องใช้
- `data/`: โฟลเดอร์เก็บข้อมูล (DB, CSV, JSON)
- `models/`: โฟลเดอร์เก็บโมเดลที่เทรนเสร็จแล้ว
- `training/`: Source code สำหรับการเทรนโมเดล
  - `prepare_dataset.py`: เตรียมข้อมูล
  - `train_classical.py`: เทรน Classical Models
  - `train_bert.py`: เทรน BERT Model
- `api/`: Source code ของ API Server
