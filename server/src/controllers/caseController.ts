import { Request, Response } from "express";
import { z } from "zod";
import { CaseModel } from "../models/Case.js";
import { CaseInput, CaseStatus } from "../types/case.js";
import { inferCaseAttributes } from "../services/modelClient.js";
import { PipelineStage } from "mongoose";

export const listCases = async (req: Request, res: Response) => {
  const { priority, status, limit, text, mode } = req.query;
  const query: Record<string, unknown> = {};

  if (priority) query.priority_label = { $in: String(priority).split(",") };
  if (status) query.status = { $in: String(status).split(",") };
  if (text) query.text = { $regex: String(text), $options: "i" };

  const parsedLimit = limit === undefined ? undefined : Number(limit);
  const cursor = CaseModel.find(query);

  if (mode === 'map') {
    cursor.select('_id location priority_label status resource_tags text updatedAt');
  } else {
    cursor.sort({ createdAt: -1 });
  }

  if (typeof parsedLimit === "number" && Number.isFinite(parsedLimit) && parsedLimit > 0) {
    cursor.limit(parsedLimit);
  }

  const cases = await cursor;
  res.json(cases);
};

export const createCase = async (
  req: Request<unknown, unknown, CaseInput>,
  res: Response
) => {
  try {
    const payload = req.body;
    console.log('Received payload:', JSON.stringify(payload));
    const enriched = await inferCaseAttributes(payload);
    console.log('Enriched data:', JSON.stringify(enriched));
    const doc = await CaseModel.create({
      source: "user_submit",
      ...enriched,
      history: [{ action: "created", by: "system" }],
    });
    res.status(201).json(doc);
  } catch (error) {
    console.error('Error creating case:', error);
    res.status(500).json({ message: 'Internal server error', error: String(error) });
  }
};

export const getCaseDetail = async (req: Request, res: Response) => {
  const doc = await CaseModel.findById(req.params.id);
  if (!doc) {
    return res.status(404).json({ message: "Case not found" });
  }
  res.json(doc);
};

const updateSchema = z.object({
  status: z.enum(["pending", "assigned", "resolved"]).optional(),
  assigned_team: z.string().optional(),
  notes: z.string().optional(),
});

export const updateCase = async (req: Request, res: Response) => {
  const parsed = updateSchema.parse(req.body);
  const doc = await CaseModel.findById(req.params.id);
  if (!doc) {
    return res.status(404).json({ message: "Case not found" });
  }

  const messages: string[] = [];
  if (parsed.status && parsed.status !== doc.status) {
    doc.status = parsed.status as CaseStatus;
    messages.push(`status -> ${parsed.status}`);
  }
  if (parsed.assigned_team && parsed.assigned_team !== doc.assigned_team) {
    doc.assigned_team = parsed.assigned_team;
    messages.push(`assigned_team -> ${parsed.assigned_team}`);
  }
  if (parsed.notes) {
    doc.notes = parsed.notes;
  }
  if (messages.length) {
    doc.history?.push({
      action: "update",
      by: req.header("x-user") || "operator",
      message: messages.join(", "),
    });
  }
  await doc.save();
  res.json(doc);
};

export const getMetricsSummary = async (_req: Request, res: Response) => {
  const [statusCounts, priorityCounts, total, tagCounts, peopleStats, medicalCount] = await Promise.all([
    CaseModel.aggregate([{ $group: { _id: "$status", total: { $sum: 1 } } }]),
    CaseModel.aggregate([
      { $group: { _id: "$priority_label", total: { $sum: 1 } } },
    ]),
    CaseModel.countDocuments(),
    CaseModel.aggregate([
      { $unwind: "$resource_tags" },
      { $group: { _id: "$resource_tags", total: { $sum: 1 } } },
      { $sort: { total: -1 } },
      { $limit: 10 }
    ]),
    CaseModel.aggregate([
      {
        $group: {
          _id: null,
          adults: { $sum: "$people.adults" },
          children: { $sum: "$people.children" },
          elderly: { $sum: "$people.elderly" },
          infants: { $sum: "$people.infants" }
        }
      }
    ]),
    CaseModel.countDocuments({ resource_tags: "medical_evac" })
  ]);

  const people = peopleStats[0] || { adults: 0, children: 0, elderly: 0, infants: 0 };
  const affectedTotal = (people.adults || 0) + (people.children || 0) + (people.elderly || 0) + (people.infants || 0);

  res.json({
    total,
    status: Object.fromEntries(
      statusCounts.map((item) => [item._id, item.total])
    ),
    priority: Object.fromEntries(
      priorityCounts.map((item) => [item._id, item.total])
    ),
    tags: Object.fromEntries(
      tagCounts.map((item) => [item._id, item.total])
    ),
    affected: affectedTotal,
    medical: medicalCount
  });
};

export const getTimelineMetrics = async (req: Request, res: Response) => {
  const days = Math.max(Number(req.query.days) || 7, 1);
  const since = new Date(Date.now() - days * 24 * 60 * 60 * 1000);

  const buckets = await CaseModel.aggregate([
    { $match: { createdAt: { $gte: since } } },
    {
      $group: {
        _id: {
          day: {
            $dateToString: {
              format: "%Y-%m-%d",
              date: "$createdAt",
              timezone: "Asia/Bangkok",
            },
          },
          priority: "$priority_label",
        },
        total: { $sum: 1 },
      },
    },
    { $sort: { "_id.day": 1 } },
  ]);

  const series: Record<
    string,
    {
      total: number;
      priority: Record<string, number>;
    }
  > = {};

  buckets.forEach((bucket) => {
    const day = bucket._id.day as string;
    const priority = bucket._id.priority as string;
    if (!series[day]) {
      series[day] = { total: 0, priority: {} };
    }
    series[day].total += bucket.total;
    series[day].priority[priority] = bucket.total;
  });

  const payload = Object.entries(series)
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([day, value]) => ({
      day,
      total: value.total,
      priority: value.priority,
    }));

  res.json(payload);
};

export const getRecentHistory = async (req: Request, res: Response) => {
  const limit = Math.min(Number(req.query.limit) || 30, 100);
  const hours = Number(req.query.hours);
  const since =
    Number.isFinite(hours) && hours > 0
      ? new Date(Date.now() - hours * 60 * 60 * 1000)
      : undefined;

  const pipeline: PipelineStage[] = [{ $unwind: "$history" }];

  if (since) {
    pipeline.push({ $match: { "history.at": { $gte: since } } });
  }

  pipeline.push(
    { $sort: { "history.at": -1 } },
    { $limit: limit },
    {
      $project: {
        case_id: "$_id",
        priority_label: 1,
        status: 1,
        text: 1,
        history: 1,
      },
    }
  );

  const rows = await CaseModel.aggregate(pipeline);

  const payload = rows.map((row) => ({
    case_id: row.case_id,
    priority_label: row.priority_label,
    status: row.status,
    text: row.text,
    entry: row.history,
  }));

  res.json(payload);
};
