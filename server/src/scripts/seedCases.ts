import fs from "node:fs";
import path from "node:path";
import { parse } from "csv-parse/sync";
import mongoose from "mongoose";
import { CaseModel } from "../models/Case.js";
import { env } from "../utils/env.js";

const DATA_PATH = path.resolve(process.cwd(), "../data/all_posts_raw.csv");

const main = async () => {
  await mongoose.connect(env.MONGO_URI);
  console.log("Connected to MongoDB");

  const content = fs.readFileSync(DATA_PATH, "utf-8");
  const rows = parse(content, { columns: true, skip_empty_lines: true });

  const docs = rows.slice(0, 500).map((row: Record<string, string>) => ({
    source: row.source || "sos_api",
    text: row.text,
    priority_label: (row.priority_label as "P1" | "P2" | "P3") ?? "P2",
    risk_flags: row.risk_flags ? JSON.parse(row.risk_flags) : {},
    resource_tags: row.resource_tags
      ? row.resource_tags.split("|").filter(Boolean)
      : [],
    location: {
      type: "Point",
      coordinates: [Number(row.lng) || 0, Number(row.lat) || 0],
    },
    address: row.location_line,
    contact: { phone: (row.phones || "").replace(/[\[\]"]/g, "") },
    context_reason: row.context_reason,
  }));

  await CaseModel.deleteMany({ source: "sos_api" });
  await CaseModel.insertMany(docs);
  console.log(`Seeded ${docs.length} cases`);
  await mongoose.disconnect();
};

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
