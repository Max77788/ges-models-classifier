// index.js
const express = require("express");
const axios = require("axios");

const app = express();
app.use(express.json());

/**
 * Call Azure Custom Vision for a batch of image URLs in parallel.
 * Returns an array: [{ imageUrl, predictions: [{ tagName, probability }, ...] }, ...]
 */
async function batchPredict(imageUrls, endpointUrl, predictionKey) {
  const requests = imageUrls.map(async (imageUrl) => {
    const resp = await axios.post(
      endpointUrl,
      { Url: imageUrl },
      {
        headers: {
          "Prediction-Key": predictionKey,
          "Content-Type": "application/json",
        },
        timeout: 25000,
      }
    );

    const predictions = resp.data.predictions || [];
    return {
      imageUrl,
      predictions: predictions.map((p) => ({
        tagName: p.tagName,
        probability: p.probability,
      })),
    };
  });

  return Promise.all(requests);
}

/**
 * Compute average probability per tagName over all images.
 * perImage = [{ imageUrl, predictions: [{ tagName, probability }, ...] }, ...]
 */
function averageByTag(perImage) {
  const sums = {};
  let count = 0;

  for (const img of perImage) {
    count += 1;
    for (const p of img.predictions) {
      if (!sums[p.tagName]) sums[p.tagName] = 0;
      sums[p.tagName] += p.probability;
    }
  }

  const averages = {};
  const n = count || 1;
  for (const [tag, sum] of Object.entries(sums)) {
    averages[tag] = sum / n;
  }

  return averages;
}

/**
 * Get label with max probability from a { [tagName]: probability } map.
 */
function getMaxLabel(probMap) {
  let bestLabel = null;
  let bestProb = -1;

  for (const [label, prob] of Object.entries(probMap)) {
    if (prob > bestProb) {
      bestProb = prob;
      bestLabel = label;
    }
  }

  return { label: bestLabel, probability: bestProb };
}

// POST /classify
// Body: { imageUrls: [ "https://...", ... ] }
app.post("/classify", async (req, res) => {
  try {
    const imageUrls = req.body.imageUrls;

    if (!Array.isArray(imageUrls) || imageUrls.length === 0) {
      return res
        .status(400)
        .json({ error: "imageUrls must be a non-empty array of URLs" });
    }

    // ---------- CONFIG (Railway envs) ----------

    const modelNotModelUrl =
      process.env.MODEL_NOT_MODEL_URL ||
      "https://gesmodelsclassification-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/35afe559-03ae-4d8c-91e9-ed001feac52e/classify/iterations/ModelNotModelPredictor/url";
    const modelNotModelKey = process.env.MODEL_NOT_MODEL_KEY;

    const civilianHybridUrl =
      process.env.CIVILIAN_HYBRID_URL ||
      "https://gesmodelsclassification-prediction.cognitiveservices.azure.com/customvision/v3.0/Prediction/d784e366-21a8-4231-8058-6f2ad50a72e0/classify/iterations/CivilianHybridPredictor/url";
    const civilianHybridKey = process.env.CIVILIAN_HYBRID_KEY;

    const superCommonUrl =
      process.env.SUPERMODEL_COMMON_URL ||
      "https://germanywestcentral.api.cognitive.microsoft.com/customvision/v3.0/Prediction/133f6dd8-c9e3-4a86-84d7-c1b6fdd4ef69/classify/iterations/SuperModelCommonModelPredictor/url";
    const superCommonKey = process.env.SUPERMODEL_COMMON_KEY;

    if (!modelNotModelKey || !civilianHybridKey || !superCommonKey) {
      return res.status(500).json({
        error:
          "Prediction keys not set (MODEL_NOT_MODEL_KEY / CIVILIAN_HYBRID_KEY / SUPERMODEL_COMMON_KEY)",
      });
    }

    // ====================================================
    // 1) STAGE 1: Model vs NOT Model (batched in parallel)
    // ====================================================

    const stage1PerImage = await batchPredict(
      imageUrls,
      modelNotModelUrl,
      modelNotModelKey
    );
    const stage1Averages = averageByTag(stage1PerImage);

    // We now know the exact tagNames from your example:
    // "Model" and "NOT Model"
    const modelProb = stage1Averages["Model"] ?? 0;
    const notModelProb = stage1Averages["NOT Model"] ?? 0;

    const isModel = modelProb >= notModelProb;

    // ====================================================
    // 2) STAGE 2: pick branch and average again
    // ====================================================

    const useSuperCommon = isModel; // if avg says Model -> super/common; else -> civilian/hybrid

    const stage2Url = useSuperCommon ? superCommonUrl : civilianHybridUrl;
    const stage2Key = useSuperCommon ? superCommonKey : civilianHybridKey;
    const predictorName = useSuperCommon
      ? "SuperModel/Common Model Predictor"
      : "Civilian/Hybrid Predictor";
    const branch = useSuperCommon ? "SuperModelCommonModel" : "CivilianHybrid";

    const stage2PerImage = await batchPredict(imageUrls, stage2Url, stage2Key);
    const stage2Averages = averageByTag(stage2PerImage);
    const finalResult = getMaxLabel(stage2Averages);

    // If you want to enforce a 0.95 threshold for “high confidence”
    const threshold = 0.95;
    const thresholdMet = finalResult.probability >= threshold;

    return res.json({
      input: {
        imageCount: imageUrls.length,
        imageUrls,
      },
      stage1: {
        predictor: "Model/NOT Model Predictor",
        averages: stage1Averages,
        perImage: stage1PerImage,
        isModel,
        modelProbability: modelProb,
        notModelProbability: notModelProb,
      },
      stage2: {
        predictor: predictorName,
        branch, // 'SuperModelCommonModel' | 'CivilianHybrid'
        averages: stage2Averages,
        perImage: stage2PerImage,
        finalLabel: finalResult.label,
        finalLabelProbability: finalResult.probability,
        highConfidence: thresholdMet,
        confidenceThreshold: threshold,
      },
    });
  } catch (err) {
    console.error(
      "ERROR /classify:",
      err?.response?.data || err.message || err
    );
    return res.status(500).json({
      error: "Internal server error",
      details: err?.response?.data || err.message || "Unknown error",
    });
  }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Classifier API listening on port ${PORT}`);
});