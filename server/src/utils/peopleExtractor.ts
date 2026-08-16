export type PeopleCounts = {
  children: number;
  elderly: number;
  adults: number;
  infants: number; // Added infants as it's in the CaseInput type
  unknown: number;
};

const PEOPLE_KEYWORDS: Record<string, string[]> = {
  children: ["เด็ก", "ลูก", "หลาน", "ทารก", "baby", "เด็กเล็ก", "เด็กน้อย", "เด็กหญิง", "เด็กชาย"],
  infants: ["ทารก", "ทารกแรกเกิด", "เด็กแรกเกิด", "แรกเกิด", "เบบี๋", "infant"],
  elderly: ["ผู้สูงอายุ", "คนแก่", "ยาย", "ตา", "อาม่า", "อากง", "คนชรา", "ผู้สูงวัย", "ลุง", "ป้า"],
  adults: ["ผู้ใหญ่", "ผู้ชาย", "ผู้หญิง", "ชาวบ้าน", "คนโต"],
};

// Regex to match "keyword number" or "keyword number คน"
// Matches: "เด็ก 2", "เด็ก 2 คน", "ผู้ใหญ่ 3", "คน 5"
const PEOPLE_PATTERN = /(เด็กเล็ก|เด็ก|ลูก|หลาน|ผู้ใหญ่|ผู้ชาย|ผู้หญิง|คนแก่|ผู้สูงอายุ|คน|ครอบครัว|ทารก|แรกเกิด|เบบี๋)\s*(?:จำนวน)?\s*(\d+)(?:\s*คน)?/g;

// Regex to match generic "number คน" if no keyword found
const GENERIC_PEOPLE_PATTERN = /(?:จำนวน)?\s*(\d+)\s*คน/g;

export const extractPeopleCounts = (text: string): PeopleCounts => {
  const counts: PeopleCounts = { children: 0, elderly: 0, adults: 0, infants: 0, unknown: 0 };
  
  if (!text) {
    return counts;
  }

  // Reset lastIndex for global regex
  PEOPLE_PATTERN.lastIndex = 0;
  GENERIC_PEOPLE_PATTERN.lastIndex = 0;

  let match;
  // First pass: look for specific keywords with numbers
  while ((match = PEOPLE_PATTERN.exec(text)) !== null) {
    const keyword = match[1];
    const value = parseInt(match[2], 10);

    if (isNaN(value) || value > 500) continue;

    let assigned = false;
    
    // Check for infants first (subset of children sometimes)
    if (PEOPLE_KEYWORDS.infants.includes(keyword)) {
      counts.infants += value;
      // Infants are also children, but usually we want to separate them if the schema allows.
      // If the schema treats them separately, we add to infants.
      // If we want total children to include infants, we might add to both, but let's keep them separate for now based on the type definition.
      assigned = true;
    } else if (PEOPLE_KEYWORDS.children.includes(keyword)) {
      counts.children += value;
      assigned = true;
    } else if (PEOPLE_KEYWORDS.elderly.includes(keyword)) {
      counts.elderly += value;
      assigned = true;
    } else if (PEOPLE_KEYWORDS.adults.includes(keyword)) {
      counts.adults += value;
      assigned = true;
    }

    if (!assigned) {
      // "คน", "ครอบครัว" -> unknown
      counts.unknown += value;
    }
  }

  // Second pass: if no unknown counts found yet, look for generic "X คน" patterns
  // This avoids double counting if "เด็ก 2 คน" was already matched by the first regex
  // However, the first regex captures "เด็ก ... 2 ... คน" so it consumes the "คน" part usually.
  // But purely "มี 5 คน" wouldn't be caught by the first regex because it requires a keyword.
  if (counts.unknown === 0 && counts.children === 0 && counts.elderly === 0 && counts.adults === 0 && counts.infants === 0) {
     while ((match = GENERIC_PEOPLE_PATTERN.exec(text)) !== null) {
        const value = parseInt(match[1], 10);
        if (!isNaN(value) && value <= 500) {
            counts.unknown += value;
        }
     }
  }

  return counts;
};
