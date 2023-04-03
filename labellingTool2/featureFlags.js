export function getFeatureFlags() {
  const mTurk = process.env.REACT_APP_MTURK === "1";
  const rate = process.env.REACT_APP_RATE === "1";
  return {
    negativeUseCases: false,
    ratePredictedUseCases: rate,
    reviewLabelling: true,
    localLabelling: !mTurk,
    mTurk: mTurk,
  };
}

export function labelUsageOptionsDynamic() {
  return {
    mTurk: false,
    negativeUseCases: false,
    ratePredictedUseCases: false,
    reviewLabelling: false,
    localLabelling: false,
    dynamicLabelling: true,
  };
}

export function localLabelling() {
  return {
    mTurk: false,
    negativeUseCases: false,
    ratePredictedUseCases: false,
    localLabelling: true,
    dynamicLabelling: false,
  };
}
