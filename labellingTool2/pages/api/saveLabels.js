import { uploadToS3, DYNAMIC_SAMPLING_BASE_FOLDER, deleteObject } from "../../utils/aws/s3";

/*
req.body = {
  run,
  sampleFileName,
  labels: 

*/
export default async function handler(req, res) {
  const sampleFileName = req.body.sampleFileName;
  const run = req.body.run;
  const key = `${DYNAMIC_SAMPLING_BASE_FOLDER}${run}/done/${sampleFileName}`;
  const data = await uploadToS3(key, JSON.stringify(req.body.labels));
  const inProgressKey = DYNAMIC_SAMPLING_BASE_FOLDER + run + '/in_progress/' + sampleFileName;
  await deleteObject(inProgressKey);
  return res.status(200).send();
};

export const config = {
  api: {
    bodyParser: {
      sizeLimit: "8mb", // Set desired value here
    },
  },
};
