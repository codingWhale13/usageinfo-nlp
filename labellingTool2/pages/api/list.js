import { listObjects } from "../../utils/s3";

export default async function handler(req, res) {
    const data = await listObjects('test');
    console.log(data);
    return data;
};