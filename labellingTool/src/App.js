import "core-js/actual/string/pad-end";
import React from 'react';
import {
  ChakraProvider,
  Box,
  VStack,
  theme,
  Grid
} from '@chakra-ui/react';
import { ColorModeSwitcher } from './ColorModeSwitcher';
import {Labeller} from './components/Labeller';
import { ReviewTokenAnnotator } from './components/ReviewTokenAnnotator';
import { Review} from './components/Review';

function App() {
  return (
    <ChakraProvider theme={theme}>
      <Box >
        <Grid minH="100vh" p={3}>
          <ColorModeSwitcher justifySelf="flex-end" />
          <VStack spacing={8}>
            <Labeller />
          </VStack>
        </Grid>
      </Box>
    </ChakraProvider>
  );
}

export default App;
