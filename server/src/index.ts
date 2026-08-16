import 'dotenv/config';
import 'express-async-errors';
import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import morgan from 'morgan';
import mongoose from 'mongoose';
import casesRouter from './routes/cases.js';
import { env } from './utils/env.js';

const app = express();

const allowedOrigins = [
  'http://localhost:5173',
  'https://helpthai.vercel.app',
  'https://helpthai-4pwt8afy0-atipongs-projects.vercel.app',
  ...(env.CORS_ORIGIN === '*' ? [] : env.CORS_ORIGIN.split(',').map((o) => o.trim()))
];

app.use(cors({
  origin: (origin, callback) => {
    if (!origin) return callback(null, true);
    if (allowedOrigins.indexOf(origin) !== -1 || env.CORS_ORIGIN === '*') {
      callback(null, true);
    } else {
      callback(new Error('Not allowed by CORS'));
    }
  }
}));
app.use(express.json({ limit: '1mb' }));
app.use(morgan('tiny'));

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', timestamp: Date.now() });
});

app.use('/api/cases', casesRouter);
app.use((err: unknown, _req: express.Request, res: express.Response, _next: express.NextFunction) => {
  console.error(err);
  res.status(500).json({ message: 'Internal server error' });
});

const run = async () => {
  await mongoose.connect(env.MONGO_URI);
  const port = Number(env.PORT) || 4000;
  app.listen(port, () => {
    console.log(`API server listening on http://localhost:${port}`);
  });
};

run().catch((err) => {
  console.error('Failed to start API server', err);
  process.exit(1);
});

