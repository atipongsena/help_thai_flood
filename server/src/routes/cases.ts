import { Router } from 'express';
import 'express-async-errors';
import {
  createCase,
  getCaseDetail,
  getMetricsSummary,
  getTimelineMetrics,
  getRecentHistory,
  listCases,
  updateCase,
} from '../controllers/caseController.js';
import { requireApiKey } from '../middleware/requireApiKey.js';

const router = Router();

router.get('/metrics/summary', getMetricsSummary);
router.get('/metrics/timeline', getTimelineMetrics);
router.get('/history/recent', getRecentHistory);
router.get('/', listCases);
router.get('/:id', getCaseDetail);
router.post('/', createCase);
router.patch('/:id', updateCase);

export default router;

