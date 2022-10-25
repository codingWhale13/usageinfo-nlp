export function CSVUpload({onUpload}){
    return <>
        <input
            type="file"
            name="file"
            accept=".tsv"
            onChange={onUpload}
            style={{ display: "block", margin: "10px auto" }}
        />
    </>
}

