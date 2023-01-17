export function downloadBlob(blob, fileName){
    const encodedUri = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', encodedUri);
    link.setAttribute('download', fileName);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

export async function parseJSONReviews(e){
  const file = e.target.files[0];
  const jsonData = JSON.parse(await file.text());
  return jsonData;
}