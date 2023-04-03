import {
  PREDICTED_USAGE_OPTIONS,
  CUSTOM_USAGE_OPTIONS,
  PREDICTED_USAGE_OPTIONS_VOTE,
  PREDICTED_USAGE_OPTION_LABEL,
  ANNOTATIONS,
} from './labelKeys';

export function downloadBlob(blob, fileName) {
  const encodedUri = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.setAttribute('href', encodedUri);
  link.setAttribute('download', fileName);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}

export function formatJsonReviews(json) {
  let reviews;
  if ('reviews' in json) {
    reviews = json['reviews'];
  } else if (Array.isArray(json)) {
    reviews = json;
  } else {
    throw new Error('Json file format not recognized');
  }

  console.log(
    reviews.map(review => 'label' in review).reduce((a, b) => a && b)
  );
  if (reviews.map(review => 'label' in review).reduce((a, b) => a && b)) {
    const annotatations = reviews.map(review =>
      review.label[ANNOTATIONS].map(annotation => {
        return {
          [PREDICTED_USAGE_OPTION_LABEL]: annotation.tokens.join(' '),
          [PREDICTED_USAGE_OPTIONS_VOTE]: NaN,
        };
      })
    );
    const customUsageOptions = reviews.map(review =>
      review.label[CUSTOM_USAGE_OPTIONS].map(usageOption => ({
        [PREDICTED_USAGE_OPTION_LABEL]: usageOption,
        [PREDICTED_USAGE_OPTIONS_VOTE]: NaN,
      }))
    );

    const predictedUsageOptions = reviews.map(
      review => review.label[PREDICTED_USAGE_OPTIONS]
    );

    const combined = annotatations.map((annotations, index) => [
      ...annotations,
      ...customUsageOptions[index],
    ]);

    predictedUsageOptions.forEach((predictedUsageOptions, index) => {
      if (predictedUsageOptions && combined[index]) {
        combined[index].forEach(usageOption => {
          if (
            predictedUsageOptions.some(
              predictedUsageOption =>
                predictedUsageOption[PREDICTED_USAGE_OPTION_LABEL] ===
                usageOption[PREDICTED_USAGE_OPTION_LABEL]
            )
          ) {
            usageOption[PREDICTED_USAGE_OPTIONS_VOTE] =
              predictedUsageOptions.find(
                predictedUsageOption =>
                  predictedUsageOption[PREDICTED_USAGE_OPTION_LABEL] ===
                  usageOption[PREDICTED_USAGE_OPTION_LABEL]
              )[PREDICTED_USAGE_OPTIONS_VOTE];
          }
        });
      }
    });

    predictedUsageOptions.forEach((predictedUsageOptions, index) => {
      if (predictedUsageOptions && combined[index]) {
        predictedUsageOptions.forEach(predictedUsageOption => {
          if (
            !combined[index].some(
              usageOption =>
                usageOption[PREDICTED_USAGE_OPTION_LABEL] ===
                predictedUsageOption[PREDICTED_USAGE_OPTION_LABEL]
            )
          ) {
            combined[index].push(predictedUsageOption);
          }
        });
      }
    });

    reviews.forEach((review, index) => {
      review.label[PREDICTED_USAGE_OPTIONS] = combined[index];
    });
  } else {
    reviews.forEach(review => {
      review.label = {
        [ANNOTATIONS]: [],
        [CUSTOM_USAGE_OPTIONS]: [],
      };
    });
  }

  return json;
}
export async function parseJSONReviews(e) {
  const file = e.target.files[0];
  const jsonData = JSON.parse(await file.text());
  return formatJsonReviews(jsonData);
}
