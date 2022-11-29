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

function App() {
  return (
    <FlagsProvider features={getFeatureFlags()}>
      <ChakraProvider theme={theme}>
        <Box >
          <Grid minH="100vh" p={3}>
            <Labeller />
          </Grid>
        </Box>
      </ChakraProvider>
    </FlagsProvider>
  );
}

export default App;
