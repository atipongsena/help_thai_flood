import { Document } from 'mongoose';

export type PriorityLabel = 'P1' | 'P2' | 'P3';
export type CaseStatus = 'pending' | 'assigned' | 'resolved';

export interface CaseInput {
  text: string;
  priority_label?: PriorityLabel;
  risk_flags?: Record<string, boolean>;
  resource_tags?: string[];
  location?: { type?: 'Point'; coordinates: [number, number] };
  address?: string;
  contact?: { name?: string; phone?: string };
  people?: { adults?: number; children?: number; elderly?: number; infants?: number };
  context_reason?: string;
  status?: CaseStatus;
  assigned_team?: string;
  notes?: string;
  history?: Array<{ action: string; by?: string; message?: string; at?: Date }>;
}

export interface CaseDocument extends Document, CaseInput {
  source: string;
  running_number?: string;
  metadata?: Record<string, unknown>;
  createdAt: Date;
  updatedAt: Date;
}

