import FileBrowser, { listFiles } from "./[...path]";

export default FileBrowser;

export async function getServerSideProps(context) {
    return await listFiles('');
}
