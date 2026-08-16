export type CaseRecord = {
  _id: string;
  text: string;
  priority_label: 'P1' | 'P2' | 'P3';
  status: 'pending' | 'assigned' | 'resolved';
  resource_tags?: string[];
  risk_flags?: Record<string, boolean>;
  location?: { type?: string; coordinates: [number, number] };
  address?: string;
  contact?: { name?: string; phone?: string };
  people?: { adults?: number; children?: number; elderly?: number; infants?: number };
  assigned_team?: string;
  notes?: string;
  history?: Array<{ action: string; message?: string; by?: string; at?: string }>;
  context_reason?: string;
  running_number?: string;
  metadata?: Record<string, unknown>;
  createdAt: string;
  updatedAt: string;
};

export type CaseMetrics = {
  total: number;
  status: Record<string, number>;
  priority: Record<string, number>;
  tags?: Record<string, number>;
  affected?: number;
  medical?: number;
};

export type CaseTimelinePoint = {
  day: string;
  total: number;
  priority: Record<string, number>;
};

export type CaseHistoryNotification = {
  case_id: string;
  priority_label: 'P1' | 'P2' | 'P3';
  status: 'pending' | 'assigned' | 'resolved';
  text: string;
  entry: { action?: string; by?: string; message?: string; at?: string };
};

