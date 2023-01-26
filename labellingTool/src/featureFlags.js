export function getFeatureFlags() {
  const mTurk = process.env.REACT_APP_MTURK === '1';
  const rate =  process.env.REACT_APP_RATE === '1';
  return {
    negativeUseCases: false,
    ratePredictedUseCases: rate,
    reviewLabelling: false,
    localLabelling: !mTurk,
    mTurk: mTurk,
  };
}
