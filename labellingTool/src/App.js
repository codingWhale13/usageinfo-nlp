import 'core-js/actual/string/pad-end';
import React from 'react';
import { ChakraProvider, Grid, Container } from '@chakra-ui/react';
import { Labeller } from './components/Labeller';
import { FlagsProvider } from 'flagged';
import { getFeatureFlags } from './featureFlags';
import MTurkReview from './components/MTurkReview';

function App() {
  const features = getFeatureFlags();
  return (
    <FlagsProvider features={getFeatureFlags()}>
      <ChakraProvider>
      <Grid minH="100vh" p={3}>

       <Container maxWidth="1300px">

        {features.mTurk ? (
            <MTurkReview />
        ) : (
            <Labeller />
        )}
        </Container>
        </Grid>
      </ChakraProvider>
    </FlagsProvider>
  );
}

export default App;
