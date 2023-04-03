import { uploadToS3 } from "../../../utils/aws/s3";
export default async function handler(req, res) {
  console.log(req);
  if (req.method === "POST") {
    const key = req.query.key.join("/");
    console.log(key);
    const data = await uploadToS3(key, req.body);
    return res.status(200).send();
  } else {
    return res.status(400).send("Upload must be a post request");
  }
}

export const config = {
  api: {
    bodyParser: {
      sizeLimit: "8mb", // Set desired value here
    },
  },
};
