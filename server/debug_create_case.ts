import 'dotenv/config';
import mongoose from 'mongoose';
import { CaseModel } from './src/models/Case.js';
import { inferCaseAttributes } from './src/services/modelClient.js';
import { env } from './src/utils/env.js';

const run = async () => {
  await mongoose.connect(env.MONGO_URI);
  
  const payload = { text: "Test case help needed" };
  console.log("Payload:", payload);
  
  try {
    const enriched = await inferCaseAttributes(payload);
    console.log("Enriched:", JSON.stringify(enriched, null, 2));
    
    const doc = await CaseModel.create({
      source: "user_submit",
      ...enriched,
      history: [{ action: "created", by: "system" }],
    });
    console.log("Created:", doc);
  } catch (err) {
    console.error("Error:", err);
  }
  
  await mongoose.disconnect();
};

run();
