import { uploadToS3, DYNAMIC_SAMPLING_BASE_FOLDER, deleteObject } from "../../utils/aws/s3";
import multiparty from "multiparty";

export default async function handler(req, res) {
    const form = new multiparty.Form();
    const data = await new Promise((resolve, reject) => {
        form.parse(req, function (err, fields, files) {
        if (err) reject({ err });
        resolve({ fields, files });
        });
    });
  console.log(`data: `, JSON.stringify(data));
    return {hello: 1};
};

export const config = {
    api: {
      bodyParser: false,
    },
  };
