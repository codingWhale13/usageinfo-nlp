export function getFeatureFlags() {
  const mTurk = process.env.REACT_APP_MTURK === '1';
  return {
    negativeUseCases: false,
    localLabelling: !mTurk,
    mTurk: mTurk,
  };
}
