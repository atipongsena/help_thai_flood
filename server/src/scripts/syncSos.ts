import fs from "node:fs";
import path from "node:path";
import mongoose from "mongoose";
import { CaseModel } from "../models/Case.js";
import { CaseInput, CaseStatus, PriorityLabel } from "../types/case.js";
import { env } from "../utils/env.js";
import { firebaseAvailable, uploadFileToFirebase } from "../utils/firebase.js";

const SOS_PATH = path.resolve(process.cwd(), "../data/sos.json");
const RAW_CSV_PATH = path.resolve(process.cwd(), "../data/all_posts_raw.csv");

type SosGeometry = {
  type?: string;
  coordinates?: [number, number];
};

type SosProperties = {
  name?: string | null;
  type_name?: string | null;
  other?: string | null;
  status_text?: string | null;
  sick_level_summary?: number | null;
};

type SosEntry = {
  _id: string;
  running_number?: string;
  location?: {
    geometry?: SosGeometry;
    properties?: SosProperties;
  };
  created_at?: string;
  updated_at?: string;
};

const RESOURCE_HINTS: Array<[RegExp, string]> = [
  [/ป่วย|หมอ|แพทย์|ไต|เจ็บ/i, "medical_evac"],
  [/อาหาร|ข้าว|น้ำ|นม/i, "food_drop"],
  [/เรือ|อพยพ|ติดอยู่|ช่วยขึ้น/i, "rescue_boat"],
  [/ไฟฟ้า|ไฟ|ไฟดับ/i, "power_supply"],
];

const sanitizeText = (value?: string | null) =>
  value?.trim() ? value.trim() : "";

const mapStatus = (status?: string | null): CaseStatus => {
  if (!status) return "pending";
  if (status.includes("รอ")) return "pending";
  if (status.includes("กำลัง")) return "assigned";
  if (status.includes("ปิด") || status.includes("เสร็จ")) return "resolved";
  return "pending";
};

const mapPriority = (
  sickLevel?: number | null,
  status?: string | null
): PriorityLabel => {
  if (sickLevel && sickLevel >= 4) return "P1";
  if (sickLevel && sickLevel >= 2) return "P2";
  if (status?.includes("รอการช่วยเหลือ")) return "P2";
  return "P3";
};

const inferResourceTags = (text: string, typeName?: string | null) => {
  if (!text && !typeName) return [];
  const target = `${text ?? ""} ${typeName ?? ""}`;
  const tags = new Set<string>();
  RESOURCE_HINTS.forEach(([pattern, tag]) => {
    if (pattern.test(target)) {
      tags.add(tag);
    }
  });
  return Array.from(tags);
};

type CasePayload = CaseInput & {
  running_number: string;
  metadata?: Record<string, unknown>;
};

const buildCasePayload = (entry: SosEntry): CasePayload => {
  const properties = entry.location?.properties;
  const geometry = entry.location?.geometry;
  const text =
    sanitizeText(properties?.other) ||
    sanitizeText(properties?.name) ||
    entry.running_number ||
    entry._id;

  const coordinates = geometry?.coordinates;
  const hasCoords = Array.isArray(coordinates) && coordinates.length === 2;
  const lat = hasCoords ? coordinates?.[1] : undefined;
  const lng = hasCoords ? coordinates?.[0] : undefined;

  const data: CasePayload = {
    source: "sos_api",
    running_number: entry.running_number || entry._id,
    text,
    priority_label: mapPriority(
      properties?.sick_level_summary ?? null,
      properties?.status_text ?? null
    ),
    status: mapStatus(properties?.status_text),
    address: sanitizeText(properties?.name) || undefined,
    context_reason: sanitizeText(properties?.other) || undefined,
    resource_tags: inferResourceTags(
      properties?.other ?? "",
      properties?.type_name
    ),
    metadata: {
      sos_id: entry._id,
      type_name: properties?.type_name,
      sick_level_summary: properties?.sick_level_summary,
      status_text: properties?.status_text,
      created_at: entry.created_at,
      updated_at: entry.updated_at,
    },
  } as Record<string, unknown>;

  if (hasCoords && typeof lat === "number" && typeof lng === "number") {
    data.location = {
      type: "Point",
      coordinates: [lng, lat],
    };
  }

  return data;
};

const readSosEntries = (): SosEntry[] => {
  if (!fs.existsSync(SOS_PATH)) {
    throw new Error(`sos.json not found at ${SOS_PATH}`);
  }
  const raw = JSON.parse(fs.readFileSync(SOS_PATH, "utf-8"));
  const entries: SosEntry[] = raw?.data?.data ?? [];
  return entries;
};

const main = async () => {
  console.log("[sync-sos] Connecting to MongoDB...");
  await mongoose.connect(env.MONGO_URI);
  const entries = readSosEntries();
  console.log(`[sync-sos] Loaded ${entries.length} entries from sos.json`);

  let inserted = 0;
  let updated = 0;
  const keepers = new Set<string>();

  for (const entry of entries) {
    const payload = buildCasePayload(entry);
    const key = (payload.running_number as string) || entry._id;
    keepers.add(key);
    const result = await CaseModel.updateOne(
      { running_number: key },
      { $set: payload },
      { upsert: true }
    );
    if (result.upsertedCount && result.upsertedCount > 0) {
      inserted += 1;
    } else {
      updated += result.modifiedCount ?? 0;
    }
  }

  const pruneResult = await CaseModel.deleteMany({
    source: "sos_api",
    running_number: { $nin: Array.from(keepers) },
  });

  console.log(
    `[sync-sos] Upserted ${inserted} new cases, updated ${updated}, removed ${pruneResult.deletedCount} stale entries`
  );

  if (firebaseAvailable) {
    const remotePath = `datasets/sos_${new Date()
      .toISOString()
      .replace(/[:.]/g, "-")}.json`;
    const uploaded = await uploadFileToFirebase(SOS_PATH, remotePath);
    if (uploaded) {
      console.log(`[sync-sos] Uploaded sos.json to ${uploaded}`);
    }
    if (fs.existsSync(RAW_CSV_PATH)) {
      const csvRemote = `datasets/all_posts_raw_${new Date()
        .toISOString()
        .replace(/[:.]/g, "-")}.csv`;
      const csvUploaded = await uploadFileToFirebase(RAW_CSV_PATH, csvRemote);
      if (csvUploaded) {
        console.log(`[sync-sos] Uploaded all_posts_raw.csv to ${csvUploaded}`);
      }
    }
  } else {
    console.log(
      "[sync-sos] Firebase config not provided, skipping cloud upload step"
    );
  }

  await mongoose.disconnect();
  console.log("[sync-sos] Done.");
};

main().catch((err) => {
  console.error("[sync-sos] Failed:", err);
  process.exit(1);
});
