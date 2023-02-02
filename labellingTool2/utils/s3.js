import { S3Client, ListObjectsCommand, GetObjectCommand, CopyObjectCommand, DeleteObjectCommand } from "@aws-sdk/client-s3";
import { PutObjectCommand } from "@aws-sdk/client-s3";

const client = new S3Client({
    region: process.env.NEXT_AWS_DEFAULT_REGION || "eu-central-1",
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

function fileToRelativeBaseDir(path, file){
    const relativePath = file.Key.replace(path, '');
    const splits = relativePath.split('/');
    if(splits.length > 1){
        return splits[0];
    }
    return null;
}
function transformFilesToDirs(path, files){
    const dirKeys = files.map((file) => fileToRelativeBaseDir(path, file));
    const uniqueDirKeys = [...new Set(dirKeys)].filter(key => key);
    return uniqueDirKeys.map((dirKey) => ({
        Key: path+dirKey,
        isDirectory: true
    }));
}

export async function listObjects(path){
    const data = await client.send(new ListObjectsCommand({
        Bucket: BUCKET,
        Prefix: path
    }));
    let files = data.Contents !== undefined ? data.Contents : [];
    const dirs = transformFilesToDirs(path, files);

    files.forEach((file) => {
        if(file.Size === 0){
            file.isDirectory = true;
        }
        else{
            file.isDirectory = false;
        }
    })

    //if the path prefix is '' for the complete bucket, only show files and dirs directory at the "root"
    files = files.filter(file => {
        const relativePath = file.Key.replace(path, '');
        const slashCount = relativePath.split('/').length;
        if(!file.isDirectory){
            return slashCount === 1;
        }
        else{
            return false;
        }
    });
    
    return [...files, ...dirs];
}

export async function getObject(key){
    // Get the object from the Amazon S3 bucket. It is returned as a ReadableStream.
    const data = await client.send(new GetObjectCommand({
        Bucket: BUCKET,
        Key: key
    }));
    // Convert the ReadableStream to a string.
    return await data;
};

export async function getAndReadStringObject(key){
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
