import mongoose from 'mongoose';
import { CaseModel } from '../models/Case.js';
import 'dotenv/config';

// Source (Local) and Target (Cloud) URIs
const LOCAL_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/flood_aid';
const CLOUD_URI = process.argv[2];

if (!CLOUD_URI) {
  console.error('Please provide the Cloud MongoDB URI as an argument.');
  console.error('Usage: npx tsx src/scripts/migrateDb.ts <CLOUD_MONGO_URI>');
  process.exit(1);
}

const migrate = async () => {
  console.log('🚀 Starting migration...');
  console.log(`From: ${LOCAL_URI}`);
  console.log(`To:   ${CLOUD_URI}`);

  try {
    // 1. Connect to Local
    console.log('Connecting to local database...');
    const localConn = await mongoose.createConnection(LOCAL_URI).asPromise();
    console.log('✅ Connected to local.');

    // 2. Read Data
    // We need to use the schema with the connection to read data
    const LocalCase = localConn.model('Case', CaseModel.schema);
    const cases = await LocalCase.find({});
    console.log(`Found ${cases.length} cases to migrate.`);

    if (cases.length === 0) {
      console.log('No data to migrate.');
      await localConn.close();
      return;
    }

    // 3. Connect to Cloud
    console.log('Connecting to cloud database...');
    const cloudConn = await mongoose.createConnection(CLOUD_URI).asPromise();
    console.log('✅ Connected to cloud.');

    const CloudCase = cloudConn.model('Case', CaseModel.schema);

    // 4. Insert Data
    console.log('Upserting data to cloud (using bulkWrite)...');
    let success = 0;
    let failed = 0;
    const BATCH_SIZE = 1000;

    for (let i = 0; i < cases.length; i += BATCH_SIZE) {
      const batch = cases.slice(i, i + BATCH_SIZE);
      const operations = batch.map((doc) => ({
        updateOne: {
          filter: { _id: doc._id },
          update: { $set: doc.toObject() },
          upsert: true,
        },
      }));

      try {
        const result = await CloudCase.bulkWrite(operations);
        success += (result.modifiedCount || 0) + (result.upsertedCount || 0);
        console.log(`Processed ${Math.min(i + BATCH_SIZE, cases.length)}/${cases.length}`);
      } catch (err) {
        console.error(`Failed to migrate batch starting at ${i}:`, err);
        failed += batch.length;
      }
    }

    console.log('-----------------------------------');
    console.log(`Migration complete!`);
    console.log(`✅ Success: ${success}`);
    console.log(`❌ Failed:  ${failed}`);

    // 5. Cleanup
    await localConn.close();
    await cloudConn.close();

  } catch (error) {
    console.error('Migration failed:', error);
    process.exit(1);
  }
};

migrate();
