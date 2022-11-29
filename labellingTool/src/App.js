import "core-js/actual/string/pad-end";
import React from 'react';
import {
  ChakraProvider,
  Box,
  VStack,
  theme,
  Grid,
  Button
} from '@chakra-ui/react';
import {Labeller} from './components/Labeller';
import { FlagsProvider } from 'flagged';
import { getFeatureFlags } from './featureFlags';
import { Review } from "./components/Review";
import MTurkReview from "./components/MTurkReview";

function App() {
  const features = getFeatureFlags();
  return (
    <FlagsProvider features={features}>
      <ChakraProvider theme={theme}>
        {features.mTurk ?
        <>
        <h1>Mturk</h1> 
        <MTurkReview />
      <crowd-button id="submitButton">Submit</crowd-button>
        </>
     

        :
        <Box >
          <Grid minH="100vh" p={3}>
            <Labeller />
          </Grid>
        </Box>
      }
      </ChakraProvider>
    </FlagsProvider>
  );
}

export default App;
