import { listObjects } from "../../utils/aws/s3";

export default async function handler(req, res) {
    const data = await listObjects('test');
    console.log(data);
    return data;
};