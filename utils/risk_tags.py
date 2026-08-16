"""
Shared keyword heuristics for risk, priority, and resource tagging.
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

RISK_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "has_children": (
        "เด็ก", "เด็กเล็ก", "เด็กน้อย", "เด็กหญิง", "เด็กชาย",
        "เด็กๆ", "ลูก", "หลาน", "เบบี๋", "baby", "child", "children",
    ),
    "has_infants": (
        "ทารก", "ทารกแรกเกิด", "เด็กแรกเกิด", "แรกเกิด",
        "5 เดือน", "6 เดือน", "3 เดือน", "เบบี๋", "infant",
    ),
    "has_elderly": (
        "ผู้สูงอายุ", "คนแก่", "ผู้สูงวัย", "คนชรา", "คุณตา", "คุณยาย",
        "ตายาย", "อาม่า", "อากง", "ลุง", "ป้า", "คนแก่พิการ",
    ),
    "has_pregnant": (
        "คนท้อง", "ท้องแก่", "ตั้งครรภ์", "แม่ท้อง", "หญิงตั้งครรภ์",
        "ใกล้คลอด", "pregnant",
    ),
    "has_bedridden": (
        "ติดเตียง", "ผู้ป่วยติดเตียง", "ให้อาหารทางสาย", "ให้อาหารผ่านท้อง",
        "ให้อาหารทางสายยาง", "ให้อาหารทางท่อ", "ให้อาหารผ่านสาย", "นอนติดเตียง",
        "สายให้อาหาร", "สายยางให้อาหาร", "สายให้น้ำเกลือ",
    ),
    "has_disabled": (
        "พิการ", "คนพิการ", "นั่งรถเข็น", "วีลแชร์", "ตาบอด",
        "หูหนวก", "down syndrome", "อัมพาต", "เดินไม่ได้",
    ),
    "has_medical": (
        "ฟอกไต", "ไตวาย", "ล้างไต", "โรคหัวใจ", "หัวใจ", "โรคไต",
        "โรคปอด", "เบาหวาน", "ความดัน", "สโตรค", "stroke",
        "dialysis", "oxygen", "ออกซิเจน", "หอบหืด", "asthma",
        "ต้องกินยา", "ยาประจำ", "ยารักษาโรค",
    ),
    "needs_medication": (
        "ยาหมด", "ยาไม่พอ", "ยาขาด", "ยาใกล้หมด", "ต้องการยา",
        "ยาความดัน", "ยาโรคหัวใจ", "ยาโรคไต", "ยาประจำตัว", "medicine",
    ),
    "needs_medical_devices": (
        "ออกซิเจน", "เครื่องออกซิเจน", "ถังออกซิเจน", "เครื่องช่วยหายใจ",
        "oxygen", "ventilator", "เครื่องผลิตออกซิเจน", "เครื่องพ่นยา",
    ),
    "has_animals": (
        "หมา", "สุนัข", "น้องหมา", "น้องแมว", "แมว", "สัตว์เลี้ยง",
        "หมู", "วัว", "ควาย", "ไก่", "เป็ด",
    ),
    "has_large_group": (
        "หลายคน", "หลายสิบคน", "หลายครอบครัว", "หลายหลังคาเรือน",
        "ทั้งซอย", "ทั้งหมู่บ้าน", "จำนวนมาก", "ร่วมร้อยคน",
        "ทั้งตึก", "ทั้งชุมชน",
    ),
    "needs_transport": (
        "ต้องการเรือ", "เรือด่วน", "เจ็ทสกี", "เจสกี", "เจ็ตสกี",
        "เรือเร็ว", "รถยกสูง", "เรือกู้ภัย", "ห้องแถวเรือ",
    ),
}

TRAP_KEYWORDS = (
    "ติดอยู่", "ติดอยู่บนหลังคา", "ติดอยู่ชั้น2", "อยู่บนหลังคา", "อยู่ดาดฟ้า",
    "ออกมาไม่ได้", "บนดาดฟ้า", "อยู่ชั้นสอง", "ชั้น2", "ชั้น 2", "ชั้นลอย",
    "ติดอยู่ในบ้าน", "ไม่มีทางออก", "ออกทางหน้าบ้านไม่ได้",
)

SUPPLY_KEYWORDS = (
    "อาหาร", "ไม่มีอาหาร", "ข้าว", "ข้าวสาร", "เสบียง",
    "น้ำ", "น้ำดื่ม", "ไม่มีน้ำ", "นม", "นมผง", "แพมเพิส",
    "ผ้าอ้อม", "อาหารสัตว์", "ของกิน", "ของใช้", "ยารักษาโรค",
    "ขาดเสบียง", "ของยังชีพ",
)

FATALITY_KEYWORDS = (
    "เสียชีวิต", "ศพ", "ผู้เสียชีวิต", "ร่าง", "จมน้ำ", "ดับ",
    "เสียชีวิตแล้ว", "ศพอยู่", "นำศพออก",
)

POWER_KEYWORDS = (
    "ไฟดับ", "ไม่มีไฟ", "ไฟฟ้าดับ", "ไฟไม่มา", "ไฟถูกตัด",
    "ไฟโดนตัด", "แบตหมด", "แบตเหลือ", "แบตจะหมด", "ชาร์จไม่ได้",
    "powerbank", "พาวเวอร์แบงก์", "เพาเวอร์แบงก์", "ไม่มีไฟชาร์จ", "ชาร์จไม่ติด",
    "เครื่องปั่นไฟ", "ไม่มีไฟส่องสว่าง",
)

COMMUNICATION_KEYWORDS = (
    "สัญญาณไม่มี", "ไม่มีสัญญาณ", "สัญญาณโทรศัพท์ไม่มี", "โทรไม่ติด",
    "ติดต่อไม่ได้", "ขาดการติดต่อ", "ไม่มีเครือข่าย", "สัญญาณไม่ดี",
    "สัญญาณขาด", "โทรศัพท์ไม่มีสัญญาณ", "เน็ตล่ม", "wifi ล่ม",
)

RISK_FLAG_NAMES = sorted(RISK_KEYWORDS.keys()) + [
    "needs_evac",
    "needs_supplies",
    "mentions_fatality",
    "needs_power",
    "needs_comms",
    "mentions_water_level",
]

HIGH_RISK_FLAG_NAMES = {
    "has_pregnant",
    "has_bedridden",
    "has_infants",
    "has_disabled",
    "needs_medical_devices",
    "needs_medication",
    "mentions_fatality",
}

VULNERABLE_FLAG_NAMES = {
    "has_children",
    "has_infants",
    "has_elderly",
    "has_medical",
    "has_disabled",
    "has_animals",
    "has_large_group",
}

RESOURCE_KEYWORDS = {
    "medical_evac": (
        "ฟอกไต", "ไตวาย", "ล้างไต", "ผู้ป่วย", "โรคหัวใจ", "ต้องไปโรงพยาบาล",
        "หายใจไม่ออก", "oxygen", "เครื่องช่วยหายใจ", "ยาหมด",
    ),
    "food_drop": (
        "อาหารหมด", "ไม่มีอาหาร", "ไม่มีน้ำ", "เสบียง", "นมผง", "แพมเพิส",
    ),
    "rescue_boat": (
        "ขอเรือ", "ขอเจ็ทสกี", "ติดหลังคา", "อพยพด่วน", "น้ำถึงชั้นสอง",
        "น้ำท่วมสูง", "เรือกู้ภัย", "เฮลิคอปเตอร์", "เรือเข้าไม่ได้",
        "น้ำไหลแรง", "รอเรือ", "ขนย้ายทางเรือ",
    ),
    "body_recovery": (
        "ศพ", "ผู้เสียชีวิต", "เก็บศพ", "รับศพ",
    ),
    "power_supply": (
        "ไฟดับ", "ไฟฟ้าถูกตัด", "ไม่มีไฟ", "powerbank", "แบตหมด",
    ),
}

RESOURCE_TAGS = tuple(RESOURCE_KEYWORDS.keys())


def _contains(text: str, keyword: str, text_lower: str) -> bool:
    if keyword.isascii():
        return keyword.lower() in text_lower
    return keyword in text


def infer_risk_flags(text: str) -> Dict[str, bool]:
    text = text or ""
    text_lower = text.lower()
    flags = {
        name: any(_contains(text, kw, text_lower) for kw in keywords)
        for name, keywords in RISK_KEYWORDS.items()
    }
    flags["needs_evac"] = any(_contains(text, kw, text_lower) for kw in TRAP_KEYWORDS)
    flags["needs_supplies"] = any(_contains(text, kw, text_lower) for kw in SUPPLY_KEYWORDS)
    flags["mentions_fatality"] = any(_contains(text, kw, text_lower) for kw in FATALITY_KEYWORDS)
    flags["needs_power"] = any(_contains(text, kw, text_lower) for kw in POWER_KEYWORDS)
    flags["needs_comms"] = any(_contains(text, kw, text_lower) for kw in COMMUNICATION_KEYWORDS)
    flags["mentions_water_level"] = (
        "น้ำท่วม" in text or "ระดับน้ำ" in text_lower or "น้ำขึ้น" in text
    )
    # Detect explicit numbers of people (>=10) to boost large group
    if not flags.get("has_large_group"):
        match = re.search(r"\b(\d{2,})\s*คน\b", text.replace(",", ""))
        if match and int(match.group(1)) >= 10:
            flags["has_large_group"] = True
    return flags


def decide_priority(urgency_score: float, flags: Dict[str, bool]) -> Tuple[str, int]:
    score = urgency_score or 0.0
    high_risk = any(flags.get(name, False) for name in HIGH_RISK_FLAG_NAMES)
    vulnerable = any(flags.get(name, False) for name in VULNERABLE_FLAG_NAMES)
    trapped = flags.get("needs_evac", False)
    infra_needs = flags.get("needs_power", False) or flags.get("needs_comms", False)

    if high_risk or (score >= 0.6 and (vulnerable or trapped)) or (trapped and vulnerable):
        return "P1", 1
    if (
        score >= 0.4
        or trapped
        or flags.get("needs_supplies", False)
        or infra_needs
        or flags.get("needs_medication", False)
        or (flags.get("has_large_group", False) and (trapped or flags.get("needs_supplies", False)))
    ):
        return "P2", 1
    return "P3", 0


def infer_resource_tags(text: str) -> List[str]:
    text = text or ""
    text_lower = text.lower()
    tags = []
    for tag, keywords in RESOURCE_KEYWORDS.items():
        if any(_contains(text, kw, text_lower) for kw in keywords):
            tags.append(tag)
    return tags


def summarize_context_reason(text: str, flags: Dict[str, bool]) -> str:
    reasons = []
    if flags.get("needs_evac"):
        reasons.append("ติดอยู่ในพื้นที่น้ำสูง/ออกไม่ได้")
    if flags.get("mentions_water_level"):
        reasons.append("น้ำท่วมลึกถึงชั้นบนหรือไหลแรง")
    if flags.get("needs_supplies"):
        reasons.append("เสบียง/น้ำอาหารหมด")
    if flags.get("needs_medication") or flags.get("has_medical"):
        reasons.append("มีผู้ป่วยต้องใช้ยาหรือรักษาต่อเนื่อง")
    if flags.get("has_pregnant") or flags.get("has_infants"):
        reasons.append("มีแม่ท้องหรือเด็กเล็กในพื้นที่เสี่ยง")
    if flags.get("needs_power"):
        reasons.append("ไฟฟ้าถูกตัด/แบตหมด")
    if flags.get("mentions_fatality"):
        reasons.append("พบผู้เสียชีวิต ต้องการการจัดการโดยด่วน")
    if not reasons:
        reasons.append("สถานการณ์ต้องติดตามเพิ่มเติม")
    return "; ".join(reasons)


PEOPLE_KEYWORDS = {
    "children": ("เด็ก", "ลูก", "หลาน", "ทารก", "baby"),
    "elderly": ("ผู้สูงอายุ", "คนแก่", "ยาย", "ตา", "อาม่า", "อากง"),
    "adults": ("ผู้ใหญ่", "ผู้ชาย", "ผู้หญิง", "ชาวบ้าน"),
}

PEOPLE_PATTERN = re.compile(r"(เด็กเล็ก|เด็ก|ลูก|หลาน|ผู้ใหญ่|ผู้ชาย|ผู้หญิง|คนแก่|ผู้สูงอายุ|คน|ครอบครัว)\s*(?:จำนวน)?\s*(\d+)(?:\s*คน)?")
GENERIC_PEOPLE_PATTERN = re.compile(r"(?:จำนวน)?\s*(\d+)\s*คน")
DURATION_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(วัน|ชั่วโมง|ชม\.|ช.ม.|hrs?|hours?)")


def extract_people_counts(text: str) -> Dict[str, int]:
    counts = {"children": 0, "elderly": 0, "adults": 0, "unknown": 0}
    if not text:
        return counts
    for match in PEOPLE_PATTERN.finditer(text):
        keyword = match.group(1)
        value = int(match.group(2))
        if value > 500:
            continue
        assigned = False
        for key, kw_list in PEOPLE_KEYWORDS.items():
            if keyword in kw_list:
                counts[key] += value
                assigned = True
                break
        if not assigned:
            counts["unknown"] += value
    if counts["unknown"] == 0:
        for match in GENERIC_PEOPLE_PATTERN.finditer(text):
            value = int(match.group(1))
            if value <= 500:
                counts["unknown"] += value
    return counts


def extract_duration_hours(text: str) -> float:
    if not text:
        return 0.0
    longest = 0.0
    for match in DURATION_PATTERN.finditer(text):
        value = float(match.group(1))
        unit = match.group(2)
        if "วัน" in unit:
            hours = value * 24.0
        else:
            hours = value
        longest = max(longest, hours)
    return longest


def serialize_flags(flags: Dict[str, bool]) -> str:
    active = [name for name, value in flags.items() if value]
    return "|".join(active)


def serialize_tags(tags: List[str]) -> str:
    return "|".join(sorted(set(tags)))


def parse_multi_label_field(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, dict):
        return [k for k, v in value.items() if v]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            data = json.loads(value)
            return parse_multi_label_field(data)
        except json.JSONDecodeError:
            return [part.strip() for part in value.split("|") if part.strip()]
    return []


