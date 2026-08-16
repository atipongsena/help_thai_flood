import admin from "firebase-admin";
import { env } from "./env.js";

let firebaseApp: admin.app.App | null = null;

const hasFirebaseConfig =
  Boolean(env.FIREBASE_PROJECT_ID) &&
  Boolean(env.FIREBASE_CLIENT_EMAIL) &&
  Boolean(env.FIREBASE_PRIVATE_KEY) &&
  Boolean(env.FIREBASE_STORAGE_BUCKET);

const initFirebase = () => {
  if (!hasFirebaseConfig) {
    return null;
  }
  if (firebaseApp) {
    return firebaseApp;
  }
  firebaseApp = admin.initializeApp({
    credential: admin.credential.cert({
      projectId: env.FIREBASE_PROJECT_ID,
      clientEmail: env.FIREBASE_CLIENT_EMAIL,
      privateKey: env.FIREBASE_PRIVATE_KEY?.replace(/\\n/g, "\n"),
    }),
    storageBucket: env.FIREBASE_STORAGE_BUCKET,
  });
  return firebaseApp;
};

export const getFirebaseApp = () => initFirebase();

export const uploadFileToFirebase = async (
  localPath: string,
  destination: string
): Promise<string | null> => {
  const app = initFirebase();
  if (!app) {
    return null;
  }
  const bucket = app.storage().bucket();
  await bucket.upload(localPath, {
    destination,
    gzip: true,
    metadata: {
      cacheControl: "public, max-age=300",
    },
  });
  const file = bucket.file(destination);
  try {
    await file.makePublic();
  } catch {
    // ignore if public ACLs are disabled
  }
  return `gs://${bucket.name}/${destination}`;
};

export const firebaseAvailable = hasFirebaseConfig;
