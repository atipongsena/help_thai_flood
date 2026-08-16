import mongoose from 'mongoose';
import { CaseDocument } from '../types/case.js';

const riskFlagsSchema = new mongoose.Schema<Record<string, boolean>>({}, { _id: false, strict: false });

const historyEntrySchema = new mongoose.Schema(
  {
    action: String,
    by: String,
    message: String,
    at: { type: Date, default: Date.now },
  },
  { _id: false }
);

const CaseSchema = new mongoose.Schema<CaseDocument>(
  {
    source: { type: String, default: 'user' },
    text: { type: String, required: true },
    priority_label: { type: String, enum: ['P1', 'P2', 'P3'], default: 'P2' },
    risk_flags: { type: riskFlagsSchema, default: {} },
    resource_tags: [{ type: String }],
    location: {
      type: { type: String, enum: ['Point'], default: 'Point' },
      coordinates: { type: [Number], index: '2dsphere', default: [0, 0] },
    },
    address: { type: String },
    contact: {
      name: String,
      phone: String,
    },
    people: {
      adults: { type: Number, default: 0 },
      children: { type: Number, default: 0 },
      elderly: { type: Number, default: 0 },
      infants: { type: Number, default: 0 },
    },
    context_reason: String,
    status: { type: String, enum: ['pending', 'assigned', 'resolved'], default: 'pending' },
    assigned_team: { type: String },
    running_number: { type: String },
    metadata: { type: mongoose.Schema.Types.Mixed },
    notes: String,
    history: { type: [historyEntrySchema], default: [] },
  },
  { timestamps: true }
);

export const CaseModel = mongoose.model<CaseDocument>('Case', CaseSchema);

