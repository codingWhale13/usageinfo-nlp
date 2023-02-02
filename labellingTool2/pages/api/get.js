import { getAndReadStringObject } from "../../utils/s3";

export default async function handler(req, res) {
    const data = await getAndReadStringObject('test/test_file');
    console.log(data);
    return data;
};