import "core-js/actual/string/pad-end";
import React from 'react';
import {
  ChakraProvider,
  Box,
  Grid,
} from '@chakra-ui/react';
import {Labeller} from './components/Labeller';
import { FlagsProvider } from 'flagged';
import { getFeatureFlags } from './featureFlags';
import { Review } from "./components/Review";
import MTurkReview from "./components/MTurkReview";

function App() {
  const features = getFeatureFlags();
  return (
    <FlagsProvider features={getFeatureFlags()}>
      <ChakraProvider>
      {features.mTurk ?
        <>
        <h1>Mturk</h1> 
        <MTurkReview />
        <crowd-button id="submitButton">Submit</crowd-button>
        </>
        :
          <Grid minH="100vh" p={3}>
            <Labeller />
          </Grid>
      }
      </ChakraProvider>
    </FlagsProvider>
  );
}

export default App;
