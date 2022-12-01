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

function App() {
  return (
    <FlagsProvider features={getFeatureFlags()}>
      <ChakraProvider>
          <Grid minH="100vh" p={3}>
            <Labeller />
          </Grid>
      </ChakraProvider>
    </FlagsProvider>
  );
}

export default App;
