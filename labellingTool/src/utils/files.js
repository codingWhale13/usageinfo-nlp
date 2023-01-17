import { PREDICTED_USAGE_OPTIONS, CUSTOM_USAGE_OPTIONS } from "./labelKeys";

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

  const reviews = jsonData['reviews'];
  /* NUR ZUM TESTEN, weil die JSON BISHER KEINE PREDICTED USAGE OPTIONS ENTHALTEN. ES WERDEN ALLE DATEN BEIM FILE EINLESEN ÜBERSCHREIBEN */
  reviews.forEach((review) => {
    review.label[PREDICTED_USAGE_OPTIONS] = review.label[CUSTOM_USAGE_OPTIONS].map((usageOption) => ({label: usageOption, vote: ''}));
  })
  return jsonData;
}