import React from 'react';
import { ChakraProvider, Container } from '@chakra-ui/react';
import { FlagsProvider } from 'flagged';
import { getFeatureFlags } from '../featureFlags';

function App() {
  const features = getFeatureFlags();
  return (
    <FlagsProvider features={getFeatureFlags()}>
      <ChakraProvider>
       <Container maxWidth="1300px" p='3'>
          gas
        </Container>
      </ChakraProvider>
    </FlagsProvider>
  );
}

export default App;