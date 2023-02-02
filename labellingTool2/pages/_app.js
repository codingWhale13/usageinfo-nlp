import { ChakraProvider, Container } from '@chakra-ui/react';
import { SessionProvider } from "next-auth/react"

function Layout({Component, pageProps: {session, ...pageProps}}) {
  return (
      <ChakraProvider>
            <SessionProvider session={session}>

       <Container maxWidth="1300px" p='3'>
            <Component {...pageProps}/>
        </Container>
        </SessionProvider>

      </ChakraProvider>
  );
}

export default Layout;