import { z } from "zod";

const envSchema = z.object({
  PORT: z.string().default("4000"),
  MONGO_URI: z.string().min(1, "MONGO_URI is required"),
  CORS_ORIGIN: z.string().default("*"),
  MODEL_API_URL: z.string().url().optional(),
  API_KEY: z.string().optional(),
  FIREBASE_PROJECT_ID: z.string().optional(),
  FIREBASE_CLIENT_EMAIL: z.string().optional(),
  FIREBASE_PRIVATE_KEY: z.string().optional(),
  FIREBASE_STORAGE_BUCKET: z.string().optional(),
});

export const env = envSchema.parse(process.env);
