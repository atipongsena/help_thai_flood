"""
Dataset Preparation Module for Thai Flood Relief NLP Pipeline

This module handles:
1. Exporting data from SQLite to CSV
2. Splitting data into Train/Test sets
3. Data augmentation (optional)
4. Label statistics and analysis
"""
import os
import sys
import sqlite3
import json
import html
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from sklearn.model_selection import train_test_split

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    DB_PATH, DATA_DIR,
    RAW_CSV_PATH, LABELED_CSV_PATH,
    TRAIN_CSV_PATH, TEST_CSV_PATH,
    SOS_JSON_PATH,
    URGENCY_KEYWORDS, TrainingConfig
)
from utils.preprocessing import calculate_urgency_score, extract_phone_numbers
from utils.risk_tags import (
    infer_risk_flags,
    infer_resource_tags,
    summarize_context_reason,
    decide_priority,
    serialize_flags,
    serialize_tags,
    extract_people_counts,
    extract_duration_hours,
)


def _ensure_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return []
    return []


def annotate_posts_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features (risk flags, resource tags, priority, counts)."""
    if df.empty:
        return df
    df = df.copy()
    df['text'] = df['text'].fillna('').astype(str)
    if 'hashtags' in df.columns:
        df['hashtags'] = df['hashtags'].apply(_ensure_list)
    else:
        df['hashtags'] = [[] for _ in range(len(df))]
    if 'phones' in df.columns:
        df['phones'] = df['phones'].apply(_ensure_list)
    else:
        df['phones'] = [[] for _ in range(len(df))]
    if 'location_line' not in df.columns:
        df['location_line'] = None
    if 'lat' not in df.columns:
        df['lat'] = None
    if 'lng' not in df.columns:
        df['lng'] = None
    df['text_length'] = df['text'].str.len()
    df['has_phone'] = df['phones'].apply(lambda x: len(x) > 0)
    df['has_location'] = df['location_line'].fillna('').astype(str).str.strip().ne('')
    df['has_coordinates'] = df['lat'].notna() & df['lng'].notna()
    df['urgency_score'] = df['text'].apply(
        lambda x: calculate_urgency_score(str(x), URGENCY_KEYWORDS)
    )
    df['risk_flags_dict'] = df['text'].apply(infer_risk_flags)
    df['risk_flags'] = df['risk_flags_dict'].apply(lambda d: json.dumps(d, ensure_ascii=False))
    df['risk_flags_active'] = df['risk_flags_dict'].apply(serialize_flags)
    df['resource_tags_list'] = df['text'].apply(infer_resource_tags)
    df['resource_tags'] = df['resource_tags_list'].apply(serialize_tags)
    df['context_reason'] = df.apply(
        lambda row: summarize_context_reason(row['text'], row['risk_flags_dict']),
        axis=1
    )
    df['priority_label'] = df.apply(
        lambda row: decide_priority(row['urgency_score'], row['risk_flags_dict'])[0],
        axis=1
    )
    df['priority_numeric'] = df['priority_label'].map({"P1": 2, "P2": 1, "P3": 0})
    people_counts = df['text'].apply(extract_people_counts)
    df['num_children'] = people_counts.apply(lambda c: c['children'])
    df['num_elderly'] = people_counts.apply(lambda c: c['elderly'])
    df['num_adults'] = people_counts.apply(lambda c: c['adults'])
    df['num_unknown_people'] = people_counts.apply(lambda c: c['unknown'])
    df['num_people_total'] = (
        df['num_children'] + df['num_elderly'] + df['num_adults'] + df['num_unknown_people']
    )
    df['duration_hours'] = df['text'].apply(extract_duration_hours)
    return df


def _decode_text(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return value
    value = html.unescape(value)
    try:
        # handle strings that were decoded with wrong codec
        return value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


TYPE_RESOURCE_HINTS = [
    ("ป่วย", "medical_evac"),
    ("หมอ", "medical_evac"),
    ("แพทย์", "medical_evac"),
    ("อาหาร", "food_drop"),
    ("น้ำ", "food_drop"),
    ("เรือ", "rescue_boat"),
    ("อพยพ", "rescue_boat"),
    ("ไฟฟ้า", "power_supply"),
    ("ไฟ", "power_supply"),
]


# =============================================================================
# Export from Database
# =============================================================================
def export_db_to_csv(
    db_path: str = DB_PATH,
    output_path: str = RAW_CSV_PATH
) -> pd.DataFrame:
    """
    Export all posts from SQLite database to CSV
    
    Args:
        db_path: Path to SQLite database
        output_path: Path for output CSV
    
    Returns:
        DataFrame with exported data
    """
    if not os.path.exists(db_path):
        print(f"Database not found: {db_path}")
        print("Please run the scraper first to collect posts.")
        return pd.DataFrame()
    
    conn = sqlite3.connect(db_path)
    
    df = pd.read_sql_query("""
        SELECT 
            id,
            source,
            url,
            text,
            hashtags,
            phones,
            lat,
            lng,
            location_line,
            created_at,
            scraped_at
        FROM posts
        ORDER BY id
    """, conn)
    
    conn.close()
    
    # Parse JSON columns
    df['hashtags'] = df['hashtags'].apply(lambda x: json.loads(x) if x else [])
    df['phones'] = df['phones'].apply(lambda x: json.loads(x) if x else [])
    
    df = annotate_posts_dataframe(df)
    
    # Add empty label column for manual annotation
    df['label'] = None
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.drop(
        columns=['risk_flags_dict', 'resource_tags_list'],
        inplace=True,
        errors='ignore'
    )
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"Exported {len(df)} posts to: {output_path}")
    print(f"\nStatistics:")
    print(f"  - With location: {df['has_location'].sum()}")
    print(f"  - With phone: {df['has_phone'].sum()}")
    print(f"  - With coordinates: {df['has_coordinates'].sum()}")
    print(f"  - Avg text length: {df['text_length'].mean():.0f} chars")
    print(f"\nPlease add labels to the 'label' column:")
    print("  0 = not urgent")
    print("  1 = urgent")
    print(f"Then save as: {LABELED_CSV_PATH}")
    
    return df


def export_sos_to_csv(
    sos_path: str = SOS_JSON_PATH,
    output_path: str = RAW_CSV_PATH
) -> pd.DataFrame:
    """
    Export SOS API JSON data into CSV compatible with the pipeline.
    """
    if not os.path.exists(sos_path):
        print(f"SOS data file not found: {sos_path}")
        print("Please provide data/sos.json first.")
        return pd.DataFrame()
    
    with open(sos_path, encoding='utf-8') as f:
        payload = json.load(f)
    
    entries = payload.get("data", {}).get("data", [])
    if not entries:
        print("No SOS records found in the JSON payload.")
        return pd.DataFrame()
    
    fetched_at = payload.get("fetched_at")
    rows = []
    for entry in entries:
        location = entry.get("location") or {}
        props = location.get("properties") or {}
        geometry = location.get("geometry") or {}
        coordinates = geometry.get("coordinates") or [None, None]
        lng, lat = None, None
        if isinstance(coordinates, (list, tuple)) and len(coordinates) >= 2:
            lng, lat = coordinates[0], coordinates[1]
        
        text_parts: List[str] = []
        other_text = props.get("other")
        if other_text:
            text_parts.append(str(html.unescape(other_text)).strip())
        fallback_bits: List[str] = []
        for key in ("type_name", "status_text"):
            val = props.get(key)
            if val:
                fallback_bits.append(str(val))
        running_number = entry.get("running_number")
        if running_number:
            fallback_bits.append(str(running_number))
        if not text_parts and fallback_bits:
            text_parts.append(" | ".join(fallback_bits))
        text = " ".join([part for part in text_parts if part]).strip()
        if not text:
            text = props.get("type_name") or props.get("status_text") or "SOS Report"
        
        hashtags = []
        raw_type_name = props.get("type_name")
        type_name = _decode_text(raw_type_name)
        if type_name:
            hashtags.append(type_name)
        status_text = _decode_text(props.get("status_text"))
        status_color = props.get("status_color")
        
        rows.append({
            "id": entry.get("_id"),
            "source": "sos_api",
            "url": f"sos://{running_number or entry.get('_id')}",
            "text": text,
            "hashtags": hashtags,
            "phones": extract_phone_numbers(text),
            "lat": lat,
            "lng": lng,
            "location_line": props.get("address") or props.get("name"),
            "created_at": entry.get("created_at"),
            "scraped_at": fetched_at or entry.get("updated_at"),
            "updated_at": entry.get("updated_at"),
            "status_text": status_text,
            "status_color": status_color,
            "status_code": props.get("status"),
            "type_name": type_name,
            "sick_level_summary": props.get("sick_level_summary"),
            "running_number": running_number,
            "raw_other": other_text,
        })
    
    df = pd.DataFrame(rows)
    df = annotate_posts_dataframe(df)
    
    # Override priority using sick level when available
    if 'sick_level_summary' in df.columns:
        sick = pd.to_numeric(df['sick_level_summary'], errors='coerce')
        df.loc[sick >= 4, 'priority_label'] = 'P1'
        df.loc[(sick >= 3) & (df['priority_label'] == 'P3'), 'priority_label'] = 'P2'
    df['priority_numeric'] = df['priority_label'].map({"P1": 2, "P2": 1, "P3": 0})
    # Enrich resource tags from type hints
    if 'resource_tags_list' in df.columns and 'type_name' in df.columns:
        def add_type_tags(row):
            tags = set(row.get('resource_tags_list') or [])
            type_text = str(row.get('type_name') or '').lower()
            for hint, tag in TYPE_RESOURCE_HINTS:
                if hint in type_text:
                    tags.add(tag)
            return sorted(tags)
        df['resource_tags_list'] = df.apply(add_type_tags, axis=1)
        df['resource_tags'] = df['resource_tags_list'].apply(serialize_tags)
    
    df['label'] = None
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.drop(
        columns=['risk_flags_dict', 'resource_tags_list'],
        inplace=True,
        errors='ignore'
    )
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"Exported {len(df)} SOS posts to: {output_path}")
    if 'type_name' in df.columns:
        type_counts = df['type_name'].fillna('unknown').value_counts().to_dict()
        print(f"Type distribution: {type_counts}")
    if 'priority_label' in df.columns:
        print(f"Priority distribution: {df['priority_label'].value_counts().to_dict()}")
    
    return df


def auto_label_posts(
    df: pd.DataFrame,
    urgency_threshold: float = 0.4
) -> pd.DataFrame:
    """
    Automatically label posts based on heuristics
    
    This is for initial labeling - should be verified manually!
    
    Args:
        df: DataFrame with posts
        urgency_threshold: Score threshold for urgent label
    
    Returns:
        DataFrame with auto-generated labels
    """
    df = df.copy()
    
    # Ensure derived columns exist
    if 'priority_label' not in df.columns or 'urgency_score' not in df.columns:
        df = annotate_posts_dataframe(df)
    
    status_text = df.get('status_text', pd.Series([""] * len(df))).fillna("").astype(str)
    sick = pd.to_numeric(df.get('sick_level_summary'), errors='coerce')
    
    df['auto_label'] = 0
    df.loc[sick >= 3, 'auto_label'] = 1
    df.loc[df['priority_label'] == 'P1', 'auto_label'] = 1
    df.loc[
        status_text.str.contains("รอการช่วยเหลือ|ขอความช่วยเหลือ|ด่วน", regex=True),
        'auto_label'
    ] = 1
    df.loc[df['urgency_score'] >= urgency_threshold, 'auto_label'] = 1
    
    # Boost urgency if location/phone present and urgency moderate
    has_contact_info = df['has_phone'] | df['has_coordinates']
    df.loc[has_contact_info & (df['urgency_score'] > 0.2), 'auto_label'] = 1
    
    print(f"\nAuto-labeling results:")
    print(df['auto_label'].value_counts())
    print(f"\nNote: These are auto-generated labels!")
    print("Please verify and adjust manually.")
    
    return df


# =============================================================================
# Train/Test Split
# =============================================================================
def split_train_test(
    labeled_path: str = LABELED_CSV_PATH,
    train_path: str = TRAIN_CSV_PATH,
    test_path: str = TEST_CSV_PATH,
    test_size: float = 0.2,
    random_state: int = 42,
    use_auto_label: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split labeled data into train and test sets
    
    Args:
        labeled_path: Path to labeled CSV
        train_path: Output path for training set
        test_path: Output path for test set
        test_size: Fraction of data for testing
        random_state: Random seed for reproducibility
        use_auto_label: Use 'auto_label' column instead of 'label'
    
    Returns:
        Tuple of (train_df, test_df)
    """
    # Auto-label workflow takes precedence when requested
    if use_auto_label:
        if os.path.exists(RAW_CSV_PATH):
            print(f"\nAuto-labeling from raw file: {RAW_CSV_PATH}")
            df = pd.read_csv(RAW_CSV_PATH)
            df = auto_label_posts(df)
            df['label'] = df['auto_label']
            df.to_csv(labeled_path, index=False, encoding='utf-8-sig')
            print(f"Saved auto-labeled file to: {labeled_path}")
        elif not os.path.exists(labeled_path):
            print("Raw file not found for auto-labeling.")
            return pd.DataFrame(), pd.DataFrame()
    else:
        if not os.path.exists(labeled_path):
            print(f"Labeled file not found: {labeled_path}")
            print("Please label the data first or use --auto-label.")
            return pd.DataFrame(), pd.DataFrame()
    
    # Load labeled data
    df = pd.read_csv(labeled_path)
    
    # Determine label column
    label_col = 'auto_label' if use_auto_label and 'auto_label' in df.columns else 'label'
    
    # Filter rows with valid labels
    df_valid = df.dropna(subset=['text', label_col])
    df_valid[label_col] = df_valid[label_col].astype(int)
    
    if len(df_valid) == 0:
        print("No labeled data found. Please add labels to the 'label' column.")
        return pd.DataFrame(), pd.DataFrame()
    
    print(f"\nDataset size: {len(df_valid)} posts")
    print(f"Label distribution:\n{df_valid[label_col].value_counts()}")
    
    # Stratified split
    try:
        train_df, test_df = train_test_split(
            df_valid,
            test_size=test_size,
            random_state=random_state,
            stratify=df_valid[label_col]
        )
    except ValueError as e:
        print(f"Warning: Could not stratify: {e}")
        train_df, test_df = train_test_split(
            df_valid,
            test_size=test_size,
            random_state=random_state
        )
    
    # Ensure label column is named 'label' in output
    if label_col != 'label':
        train_df['label'] = train_df[label_col]
        test_df['label'] = test_df[label_col]
    
    # Save splits
    os.makedirs(os.path.dirname(train_path) if os.path.dirname(train_path) else '.', exist_ok=True)
    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    test_df.to_csv(test_path, index=False, encoding='utf-8-sig')
    
    print(f"\nSaved:")
    print(f"  Train: {train_path} ({len(train_df)} samples)")
    print(f"  Test:  {test_path} ({len(test_df)} samples)")
    
    return train_df, test_df


# =============================================================================
# Data Analysis
# =============================================================================
def analyze_dataset(df: pd.DataFrame) -> dict:
    """
    Analyze dataset and return statistics
    
    Args:
        df: DataFrame with posts
    
    Returns:
        Dict with statistics
    """
    stats = {
        'total_posts': len(df),
        'text_length_mean': df['text'].str.len().mean(),
        'text_length_std': df['text'].str.len().std(),
        'posts_with_location': df['location_line'].notna().sum() if 'location_line' in df.columns else 0,
        'posts_with_phone': df['phones'].apply(lambda x: len(json.loads(x) if isinstance(x, str) else x) > 0).sum() if 'phones' in df.columns else 0,
        'posts_with_coords': (df['lat'].notna() & df['lng'].notna()).sum() if 'lat' in df.columns else 0,
    }
    
    if 'label' in df.columns:
        label_counts = df['label'].value_counts().to_dict()
        stats['label_distribution'] = label_counts
    
    return stats


def print_dataset_stats(train_path: str = TRAIN_CSV_PATH, test_path: str = TEST_CSV_PATH):
    """Print statistics for train and test datasets"""
    
    for name, path in [("Train", train_path), ("Test", test_path)]:
        if not os.path.exists(path):
            print(f"{name} file not found: {path}")
            continue
        
        df = pd.read_csv(path)
        print(f"\n=== {name} Dataset ===")
        print(f"Total samples: {len(df)}")
        
        if 'label' in df.columns:
            print(f"Label distribution:")
            print(df['label'].value_counts())
        
        print(f"Avg text length: {df['text'].str.len().mean():.0f} chars")


# =============================================================================
# Sample Data Generation (for testing)
# =============================================================================
def create_sample_data(output_path: str = None, n_samples: int = 20):
    """
    Create sample flood relief posts for testing
    
    Args:
        output_path: Path to save sample data
        n_samples: Number of samples to create
    """
    sample_posts = [
        # Urgent posts
        {
            "text": "🆘 ขออพยพด่วน!!! ผู้ประสบภัย 3 คน คนท้องแก่ ผู้สูงอายุพิการ 80 ปี อยู่บนหลังคา น้ำท่วมมิดบ้าน 2 วันแล้ว พิกัด 7.0074, 100.4407 โทร 0819797123",
            "label": 1
        },
        {
            "text": "ช่วยด้วยครับ น้ำเข้าบ้านแล้ว มีเด็กเล็ก 2 คน ที่อยู่ หมู่บ้านพฤกษา ซอย 5 ต.หาดใหญ่ อ.หาดใหญ่ จ.สงขลา ติดต่อ 0891234567",
            "label": 1
        },
        {
            "text": "SOS ครอบครัวติดอยู่ชั้น 2 น้ำเริ่มขึ้นสูง ไฟดับ แบตโทรศัพท์ใกล้หมด บ้านเลขที่ 168 ถนนสุทธิสมิทธิ์ โทรด่วน 0867891234",
            "label": 1
        },
        {
            "text": "ต้องการความช่วยเหลือเร่งด่วน ผู้ป่วยติดเตียง น้ำท่วมถึงระดับเอว ไม่สามารถเคลื่อนย้ายเองได้ ต.ควนลัง อ.หาดใหญ่ 0845678901",
            "label": 1
        },
        {
            "text": "#ขอความช่วยเหลือ น้ำท่วมบ้าน 5 ครอบครัว รวม 15 คน มีผู้สูงอายุ 4 คน ต้องการอพยพ หมู่ 3 ต.คลองแห อ.หาดใหญ่ 0823456789",
            "label": 1
        },
        {
            "text": "ด่วนมาก! คนจมน้ำ รอความช่วยเหลือ พิกัด 7.0234, 100.4567 โทร 0812345678 มาเร็วที่สุด!",
            "label": 1
        },
        {
            "text": "ขอความช่วยเหลือด่วน มีคนป่วยต้องล้างไต น้ำท่วมไม่สามารถไปโรงพยาบาลได้ ต้องการเรือ ที่อยู่ ซอยพัฒนา 3 ต.บ่อยาง",
            "label": 1
        },
        {
            "text": "#น้ำท่วม68 บ้านติดอยู่ 3 วันแล้ว อาหารหมด น้ำดื่มหมด มีเด็กทารก 1 คน ช่วยด้วยค่ะ 0876543210",
            "label": 1
        },
        {
            "text": "ขอเรือด่วน! คนแก่ไม่สามารถเดินได้ น้ำสูงมาก ต้องอพยพ บ้านเลขที่ 45 หมู่ 7 ต.พะตง อ.หาดใหญ่ โทร 0854321098",
            "label": 1
        },
        {
            "text": "🆘 ติดอยู่บนหลังคา 2 คน รอกู้ภัยมาช่วย น้ำไหลแรงมาก หมู่บ้านการเคหะ จ.สงขลา 0898765432",
            "label": 1
        },
        
        # Non-urgent posts
        {
            "text": "น้ำท่วมหาดใหญ่หนักมาก ทุกคนระวังตัวด้วยนะครับ #น้ำท่วม68",
            "label": 0
        },
        {
            "text": "รายงานสถานการณ์น้ำท่วมภาคใต้ ฝนยังตกต่อเนื่อง คาดว่าจะดีขึ้นในอีก 2-3 วัน",
            "label": 0
        },
        {
            "text": "ขอบคุณทีมกู้ภัยที่มาช่วยเหลือครับ ปลอดภัยแล้ว #ขอความช่วยเหลือ #น้ำท่วม",
            "label": 0
        },
        {
            "text": "แชร์ให้ด้วยนะครับ ใครต้องการความช่วยเหลือ โทรสายด่วน 1784 หรือ 199",
            "label": 0
        },
        {
            "text": "ประกาศ: ศูนย์พักพิงผู้ประสบภัยน้ำท่วม เปิดรับผู้อพยพที่โรงเรียนหาดใหญ่วิทยาลัย",
            "label": 0
        },
        {
            "text": "ระดับน้ำในคลองอู่ตะเภาเริ่มลดลงแล้ว คาดว่าอีก 6 ชั่วโมงจะกลับสู่ปกติ",
            "label": 0
        },
        {
            "text": "รวมเบอร์โทรหน่วยกู้ภัย: กู้ภัยหาดใหญ่ 074-xxxxxx, มูลนิธิร่วมกตัญญู 1669",
            "label": 0
        },
        {
            "text": "#น้ำท่วมหาดใหญ่ ถนนเพชรเกษมผ่านได้แล้ว รถเล็กระวังน้ำรอระบาย",
            "label": 0
        },
        {
            "text": "บริจาคสิ่งของช่วยผู้ประสบภัยได้ที่ศาลากลางจังหวัดสงขลา เปิดรับทุกวัน 8:00-18:00",
            "label": 0
        },
        {
            "text": "สถานการณ์น้ำท่วมในพื้นที่อำเภอเมืองสงขลาเริ่มคลี่คลาย ชาวบ้านเริ่มทำความสะอาดบ้านเรือน",
            "label": 0
        },
    ]
    
    # Create DataFrame
    df = pd.DataFrame(sample_posts[:n_samples])
    
    # Add synthetic metadata
    df['id'] = range(1, len(df) + 1)
    df['source'] = 'sample'
    df['url'] = df['id'].apply(lambda x: f'sample://{x}')
    df['hashtags'] = df['text'].apply(
        lambda x: [tag for tag in ['น้ำท่วม68', 'ขอความช่วยเหลือ'] if tag in x]
    )

    def random_phone_list():
        return ['0812345678'] if np.random.rand() < 0.5 else []

    df['phones'] = df['text'].apply(lambda _: random_phone_list())
    df['lat'] = None
    df['lng'] = None
    df['location_line'] = None
    df['has_location'] = False
    df['has_phone'] = df['phones'].apply(lambda x: len(x) > 0)
    df['has_coordinates'] = False
    df['urgency_score'] = df['text'].apply(
        lambda x: calculate_urgency_score(x, URGENCY_KEYWORDS)
    )
    
    if output_path is None:
        output_path = LABELED_CSV_PATH
    
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"Created {len(df)} sample posts at: {output_path}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    
    return df


# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare dataset for training")
    parser.add_argument("--export", action="store_true",
                       help="Export raw data to CSV for labeling")
    parser.add_argument("--split", action="store_true",
                       help="Split labeled data into Train/Test")
    parser.add_argument("--sample", action="store_true",
                       help="Create sample data for testing")
    parser.add_argument("--auto-label", action="store_true",
                       help="Use auto-labeling for split")
    parser.add_argument("--stats", action="store_true",
                       help="Print dataset statistics")
    parser.add_argument("--sos", action="store_true",
                       help="Use SOS API data as the primary source")
    
    args = parser.parse_args()
    
    if args.sample:
        create_sample_data()
        split_train_test(use_auto_label=False)
    elif args.export:
        if args.sos:
            export_sos_to_csv()
        else:
            export_db_to_csv()
    elif args.split:
        split_train_test(use_auto_label=args.auto_label)
    elif args.stats:
        print_dataset_stats()
    else:
        # Default: prefer SOS data when available
        exported = False
        if os.path.exists(SOS_JSON_PATH):
            print("Exporting SOS data...")
            export_sos_to_csv()
            exported = True
        elif os.path.exists(DB_PATH):
            print("Exporting database...")
            export_db_to_csv()
            exported = True
        
        if exported and (os.path.exists(LABELED_CSV_PATH) or os.path.exists(RAW_CSV_PATH)):
            print("\nSplitting data...")
            split_train_test(use_auto_label=True)
        elif not exported:
            print("\nNo data found. Creating sample data...")
            create_sample_data()
            split_train_test()

