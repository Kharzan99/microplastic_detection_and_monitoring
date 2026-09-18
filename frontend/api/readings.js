const { initializeApp, cert, getApps } = require("firebase-admin/app");
const { getDatabase } = require("firebase-admin/database");

function getFirebaseDatabase() {
  if (getApps().length > 0) {
    return getDatabase();
  }

  const serviceAccountJson = process.env.FIREBASE_SERVICE_ACCOUNT_JSON;
  const databaseURL = process.env.FIREBASE_DATABASE_URL;

  if (!serviceAccountJson) {
    throw new Error("FIREBASE_SERVICE_ACCOUNT_JSON is not configured");
  }

  if (!databaseURL) {
    throw new Error("FIREBASE_DATABASE_URL is not configured");
  }

  let serviceAccount;

  try {
    serviceAccount = JSON.parse(serviceAccountJson);
  } catch (error) {
    throw new Error("FIREBASE_SERVICE_ACCOUNT_JSON is not valid JSON");
  }

  initializeApp({
    credential: cert(serviceAccount),
    databaseURL,
  });

  return getDatabase();
}

function normalizeReadings(raw) {
  if (!raw || typeof raw !== "object") {
    return [];
  }

  return Object.entries(raw)
    .map(([key, value]) => {
      if (!value || typeof value !== "object") {
        return null;
      }

      return {
        ...value,
        _key: key,
      };
    })
    .filter(Boolean);
}

function getLatestRunId(readings) {
  const tagged = readings.filter(
    (item) => item.analysis_run_id
  );

  if (!tagged.length) {
    return null;
  }

  tagged.sort((a, b) => {
    const aTime = Number(
      a.run_started_timestamp || a.timestamp || 0
    );

    const bTime = Number(
      b.run_started_timestamp || b.timestamp || 0
    );

    return bTime - aTime;
  });

  return tagged[0].analysis_run_id;
}

module.exports = async function handler(req, res) {
  if (req.method !== "GET") {
    return res.status(405).json({
      ok: false,
      error: "Method not allowed",
    });
  }

  try {
    const database = getFirebaseDatabase();

    const snapshot = await database
      .ref("sensor_readings")
      .once("value");

    const raw = snapshot.val();
    const allReadings = normalizeReadings(raw);

    /*
     * The local main_demo.py gives every execution one
     * analysis_run_id.
     *
     * For the public Vercel dashboard we expose only the
     * newest tagged run, rather than the entire Firebase history.
     */
    const latestRunId = getLatestRunId(allReadings);

    if (!latestRunId) {
      res.setHeader("Cache-Control", "no-store");

      return res.status(200).json({
        ok: true,
        source: "firebase",
        current_run_id: null,
        current_run_count: 0,
        current_run_mode: null,
        current_run_started_human: null,
        readings: [],
      });
    }

    const currentRun = allReadings
      .filter(
        (item) => item.analysis_run_id === latestRunId
      )
      .sort(
        (a, b) =>
          Number(b.timestamp || 0) -
          Number(a.timestamp || 0)
      );

    const first = currentRun[0] || {};

    res.setHeader("Cache-Control", "no-store");

    return res.status(200).json({
      ok: true,
      source: "firebase",
      current_run_id: latestRunId,
      current_run_count: currentRun.length,
      current_run_mode: first.run_mode || first.mode || null,
      current_run_started_human:
        first.run_started_human || null,
      readings: currentRun,
    });
  } catch (error) {
    console.error("AquaTrace Firebase API error:", error);

    return res.status(500).json({
      ok: false,
      error: error.message || "Firebase API error",
    });
  }
};