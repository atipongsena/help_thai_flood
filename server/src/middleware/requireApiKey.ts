import { Request, Response, NextFunction } from 'express';
import { env } from '../utils/env.js';

export const requireApiKey = (req: Request, res: Response, next: NextFunction) => {
  if (!env.API_KEY) {
    return next();
  }
  const headerKey = req.header('x-api-key');
  if (headerKey && headerKey === env.API_KEY) {
    return next();
  }
  return res.status(401).json({ message: 'Unauthorized' });
};

