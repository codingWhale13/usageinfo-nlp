import { listObjects, moveObject, DYNAMIC_SAMPLING_BASE_FOLDER } from "../../utils/s3";
import { dateToString } from "../../utils/serializeToJson";
import { File } from "../../components/FileBrowser/File";
import { Dir } from "../../components/FileBrowser/Dir";
import { Text } from "@chakra-ui/react";
export function FileBrowser({ files, dirs, ...otherProps }) {
    if (files.length === 0 && dirs.length === 0) {
        return <Text>No files found. Are you sure the path is correct?</Text>
    }
    return (
        <>
            {dirs.map((dir) =>
                <Dir {...dir} />
            )}
            {files.map((file) =>
                <File {...file} />
            )}
        </>
    );
}

export async function listFiles(path) {
    const contents = (await listObjects(path));
    console.log('path',path, contents.length);
    dateToString(contents);
    const dirs = contents.filter(object => object.isDirectory);
    const files = contents.filter(object => !object.isDirectory);
    return { props: { files: files, dirs: dirs } };
}

export async function getServerSideProps(context) {
    const path = context.params.path.join('/') + '/';
    return await listFiles(path);
}

export default FileBrowser;