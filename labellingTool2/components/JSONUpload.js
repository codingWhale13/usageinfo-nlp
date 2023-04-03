export function JSONUpload({ onUpload }) {
  return (
    <>
      <input
        type="file"
        name="file"
        accept=".json"
        onChange={onUpload}
        style={{ display: "block", margin: "10px auto" }}
      />
    </>
  );
}
