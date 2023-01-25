export function getFeatureFlags() {
  const mTurk = process.env.REACT_APP_MTURK === '1';
  return {
    negativeUseCases: false,
    ratePredictedUseCases: true,
    reviewLabelling: false,
    localLabelling: false,
    mTurk: mTurk,
  };
}
