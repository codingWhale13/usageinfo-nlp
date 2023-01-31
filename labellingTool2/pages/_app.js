import { ChakraProvider, Container } from '@chakra-ui/react';

function Layout({Component, pageProps}) {
  return (

      <ChakraProvider>
       <Container maxWidth="1300px" p='3'>
            <Component {...pageProps}/>
        </Container>
      </ChakraProvider>
  );
}

export default Layout;