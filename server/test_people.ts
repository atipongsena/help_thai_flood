// Mock environment variables before imports
process.env.MONGO_URI = "mongodb://localhost:27017/test";
process.env.PORT = "4000";
process.env.MODEL_API_URL = "http://localhost:5000"; // Optional but good to set

const runTests = async () => {
  // Dynamic import to ensure env vars are set before module load
  const { inferCaseAttributes } = await import('./src/services/modelClient.js');
  const { CaseInput } = await import('./src/types/case.js');

  console.log("Running People Extraction Tests...\n");

  const testCases = [
    "ช่วยด้วย น้ำท่วม มีเด็ก 2 คน คนแก่ 1 คน",
    "มีผู้ป่วยติดเตียง 1 คน และเด็กเล็ก 3 คน",
    "ครอบครัว 5 คน ติดบนหลังคา",
    "ไม่มีใครอยู่",
  ];

  for (const text of testCases) {
    const payload = { text }; // Simple object, type check later
    const result = await inferCaseAttributes(payload as any);
    console.log(`Text: "${text}"`);
    console.log("Extracted People:", JSON.stringify(result.people, null, 2));
    console.log("-".repeat(40));
  }
};

runTests();
