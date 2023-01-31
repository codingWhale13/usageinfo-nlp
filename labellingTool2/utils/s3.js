import { S3Client, ListObjectsCommand, GetObjectCommand, CopyObjectCommand, DeleteObjectCommand } from "@aws-sdk/client-s3";
import { PutObjectCommand } from "@aws-sdk/client-s3";

console.log( process.env.AWS_DEFAULT_REGION, process.env.AWS_ACCESS_KEY_ID, process.env.AWS_SECRET_ACCESS_KEY);
const client = new S3Client({
    region: "eu-central-1",
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
});

const BUCKET = 'bsc2022-usageinfo';

export async function uploadToS3(key, body){
    const fileParams = {
        Bucket: BUCKET,
        Key: key,
        Body: body
    };

    const data = await client.send(new PutObjectCommand(fileParams));
    return data;
}


export async function listObjects(path){
    const data = await client.send(new ListObjectsCommand({
        Bucket: BUCKET,
        Prefix: path
    }));
    console.log("Success", data);
    return data; // For unit tests.
}

export async function getObject(key){
    // Get the object from the Amazon S3 bucket. It is returned as a ReadableStream.
    const data = await client.send(new GetObjectCommand({
        Bucket: BUCKET,
        Key: key
    }));
    // Convert the ReadableStream to a string.
    return await data.Body.transformToString();
};


export async function moveObject(sourceKey, targetKey){
    const dataCopy = await client.send(new CopyObjectCommand({
        Bucket: BUCKET,
        CopySource: `/${BUCKET}/${sourceKey}`,
        Key: targetKey
    }));
    
    const dataDelete = await client.send(new DeleteObjectCommand({
        Bucket: BUCKET,
        Key: sourceKey,
    }));
    

    return {copy: dataCopy, delete: undefined};
}

export async function deleteObject(key){
    const data = await client.send(new DeleteObjectCommand({
        Bucket: BUCKET,
        Key: key
    }));
    return data;
}

export const DYNAMIC_SAMPLING_BASE_FOLDER = 'dynamic_samples/';
