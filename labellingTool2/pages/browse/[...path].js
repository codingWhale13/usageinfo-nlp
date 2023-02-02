import { listObjects, moveObject, DYNAMIC_SAMPLING_BASE_FOLDER } from "../../utils/s3";
import { dateToString } from "../../utils/serializeToJson";
import { File } from "../../components/FileBrowser/File";
import { Dir } from "../../components/FileBrowser/Dir";
import { Text } from "@chakra-ui/react";
import Dropzone from 'react-dropzone';
import { StyledDropzone } from "../../components/FileBrowser/Upload";

export function FileBrowser({ files, dirs, path, ...otherProps }) {
    if (files.length === 0 && dirs.length === 0) {
        return <Text>No files found. Are you sure the path is correct?</Text>
    }
    return (
        <>
            {dirs.map((dir) =>
                <Dir {...dir} key={dir.Key} />
            )}
            {files.map((file) =>
                <File {...file} key={file.Key} />
            )}


            <StyledDropzone onDrop={acceptedFiles => {
                acceptedFiles.forEach((file) => {
                    const reader = new FileReader()

                    reader.onabort = () => console.log('file reading was aborted')
                    reader.onerror = () => console.log('file reading has failed')
                    reader.onload = async () => {
                        // Do whatever you want with the file contents
                        const binaryStr = reader.result
                        console.log(file, binaryStr);

                        const res = await fetch('/api/upload/'+path+file.name, {
                            method: 'POST',
                            body: binaryStr
                        });

                        window.location.reload();
                    }
                    reader.readAsArrayBuffer(file)
                })

            }} />
        </>
    );
}

export async function listFiles(path) {
    const contents = (await listObjects(path));
    console.log('path', path, contents.length);
    dateToString(contents);
    const dirs = contents.filter(object => object.isDirectory);
    const files = contents.filter(object => !object.isDirectory);
    return { props: { files: files, dirs: dirs, path: path } };
}

export async function getServerSideProps(context) {
    const path = context.params.path.join('/') + '/';
    return await listFiles(path);
}

export default FileBrowser;