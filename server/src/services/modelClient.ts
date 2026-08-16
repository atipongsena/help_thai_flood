import fetch from 'node-fetch';
import { env } from '../utils/env.js';
import { CaseInput } from '../types/case.js';
import { extractPeopleCounts } from '../utils/peopleExtractor.js';

type ModelResponse = {
  priority: { label: string; scores: Record<string, number> };
  risk_flags: Record<string, number>;
  resource_tags: Record<string, number>;
  context_reason?: string;
};

const threshold = 0.5;

export const inferCaseAttributes = async (payload: CaseInput): Promise<CaseInput> => {
  const peopleCounts = extractPeopleCounts(payload.text);
  
  const mergedPeople = {
    adults: (payload.people?.adults || 0) + peopleCounts.adults,
    children: (payload.people?.children || 0) + peopleCounts.children,
    elderly: (payload.people?.elderly || 0) + peopleCounts.elderly,
    infants: (payload.people?.infants || 0) + peopleCounts.infants,
  };
  
  mergedPeople.adults += peopleCounts.unknown;

  if (!env.MODEL_API_URL) {
    return {
      ...payload,
      people: mergedPeople,
    };
  }

  try {
    const response = await fetch(env.MODEL_API_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: payload.text }),
    });

    if (!response.ok) {
      throw new Error(`Model API error: ${response.status}`);
    }

    const data = (await response.json()) as ModelResponse;
    return {
      ...payload,
      priority_label: (data.priority?.label as CaseInput['priority_label']) ?? payload.priority_label,
      risk_flags: Object.fromEntries(
        Object.entries(data.risk_flags || {}).map(([name, score]) => [name, score >= threshold])
      ),
      resource_tags: Object.entries(data.resource_tags || {})
        .filter(([, score]) => score >= threshold)
        .map(([name]) => name),
      context_reason: data.context_reason ?? payload.context_reason,
      people: mergedPeople,
    };
  } catch (error) {
    console.error('Model inference failed, falling back to payload', error);
    return {
      ...payload,
      people: mergedPeople,
    };
  }
};

