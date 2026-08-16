import 'dotenv/config';
import mongoose from 'mongoose';
import { CaseModel } from '../src/models/Case.js';
import { extractPeopleCounts } from '../src/utils/peopleExtractor.js';
import { env } from '../src/utils/env.js';

const run = async () => {
  console.log('Connecting to MongoDB...');
  // Ensure MONGO_URI is available
  if (!env.MONGO_URI) {
    console.error('MONGO_URI is not defined in environment variables.');
    process.exit(1);
  }
  
  await mongoose.connect(env.MONGO_URI);
  console.log('Connected to MongoDB.');

  console.log('Fetching all cases...');
  const cases = await CaseModel.find({});
  console.log(`Found ${cases.length} cases.`);

  let updatedCount = 0;
  let processedCount = 0;

  for (const doc of cases) {
    processedCount++;
    const counts = extractPeopleCounts(doc.text);
    
    // Calculate total found
    const totalFound = counts.adults + counts.children + counts.elderly + counts.infants + counts.unknown;
    
    if (totalFound > 0) {
        // Prepare new people object
        // We map 'unknown' to 'adults' to ensure they are counted
        const newPeople = {
            adults: counts.adults + counts.unknown,
            children: counts.children,
            elderly: counts.elderly,
            infants: counts.infants
        };

        // Check if update is needed (simple check if current is all 0 and we found something, 
        // or just overwrite to be sure we have the latest extraction logic)
        // Let's overwrite to ensure consistency with the new logic.
        
        doc.people = newPeople;
        
        // We don't want to trigger a full validation or hooks if not needed, but save() is fine.
        await doc.save();
        updatedCount++;
    }
    
    if (processedCount % 100 === 0) {
        process.stdout.write(`Processed ${processedCount}/${cases.length}...\r`);
    }
  }
  
  console.log(`\nFinished! Updated ${updatedCount} cases out of ${cases.length}.`);
  await mongoose.disconnect();
};

run().catch((err) => {
  console.error('Error running backfill:', err);
  process.exit(1);
});
