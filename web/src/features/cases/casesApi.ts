import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';
import { CaseHistoryNotification, CaseMetrics, CaseRecord, CaseTimelinePoint } from '../../types/case';

type ListFilters = { priority: string[]; status: string[]; mode?: 'map' };

export const casesApi = createApi({
  reducerPath: 'casesApi',
  baseQuery: fetchBaseQuery({ baseUrl: 'https://flood-server-sduwialsmq-as.a.run.app/api/' }),
  tagTypes: ['Case'],
  endpoints: (builder) => ({
    getCases: builder.query<CaseRecord[], ListFilters>({
      query: (filters) => {
        const params = new URLSearchParams();
        if (filters.priority.length) params.set('priority', filters.priority.join(','));
        if (filters.status.length) params.set('status', filters.status.join(','));
        if (filters.mode) params.set('mode', filters.mode);
        params.set('limit', '0');
        return `cases?${params.toString()}`;
      },
      providesTags: ['Case'],
    }),
    getCaseDetail: builder.query<CaseRecord, string>({
      query: (id) => `cases/${id}`,
      providesTags: (_result, _err, id) => [{ type: 'Case', id }],
    }),
    createCase: builder.mutation<CaseRecord, Partial<CaseRecord>>({
      query: (body) => ({
        url: 'cases',
        method: 'POST',
        body,
      }),
      invalidatesTags: ['Case'],
    }),
    updateCase: builder.mutation<CaseRecord, { id: string; patch: Partial<CaseRecord> }>({
      query: ({ id, patch }) => ({
        url: `cases/${id}`,
        method: 'PATCH',
        body: patch,
      }),
      invalidatesTags: (_res, _err, arg) => [{ type: 'Case', id: arg.id }, 'Case'],
    }),
    getMetrics: builder.query<CaseMetrics, void>({
      query: () => 'cases/metrics/summary',
      providesTags: ['Case'],
    }),
    getTimeline: builder.query<CaseTimelinePoint[], number | void>({
      query: (days = 7) => `cases/metrics/timeline?days=${days}`,
    }),
    getNotifications: builder.query<CaseHistoryNotification[], { limit?: number } | void>({
      query: (params) => {
        const search = new URLSearchParams();
        if (params?.limit) search.set('limit', String(params.limit));
        return `cases/history/recent?${search.toString()}`;
      },
    }),
    getBoardCases: builder.query<CaseRecord[], number | void>({
      query: (limit = 120) => `cases?limit=${limit}`,
      providesTags: ['Case'],
    }),
  }),
});

export const {
  useGetCasesQuery,
  useGetCaseDetailQuery,
  useCreateCaseMutation,
  useUpdateCaseMutation,
  useGetMetricsQuery,
  useGetTimelineQuery,
  useGetNotificationsQuery,
  useGetBoardCasesQuery,
} = casesApi;

