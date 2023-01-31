import 'core-js/actual/string/pad-end';
import React from 'react';
import { ChakraProvider, Container } from '@chakra-ui/react';
import { Labeller } from '../components/Labeller';
import { FlagsProvider } from 'flagged';
import { getFeatureFlags } from '../featureFlags';
import MTurkReview from '../components/MTurkReview';

function App() {
  const features = getFeatureFlags();
  return (
    <FlagsProvider features={getFeatureFlags()}>
      <ChakraProvider>
       <Container maxWidth="1300px" p='3'>
        {features.mTurk ? (
            <MTurkReview />
        ) : (
            <Labeller />
        )}
        </Container>
      </ChakraProvider>
    </FlagsProvider>
  );
}

export default App;