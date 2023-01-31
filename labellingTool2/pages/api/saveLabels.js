import { uploadToS3, DYNAMIC_SAMPLING_BASE_FOLDER, deleteObject } from "../../utils/s3";

export default async function handler(req, res) {
    const sampleFileName = req.body.sampleFileName;
    const run = req.body.run;
    const key = `${DYNAMIC_SAMPLING_BASE_FOLDER}${run}/done/${sampleFileName}`;
    const data = await uploadToS3(key, JSON.stringify(req.body.labels));
    const inProgressKey = DYNAMIC_SAMPLING_BASE_FOLDER + run + '/in_progress/' + sampleFileName;
    await deleteObject(inProgressKey);
    return data;
};

export const config = {
  api: {
    bodyParser: {
      sizeLimit: "8mb", // Set desired value here
    },
  },
};
