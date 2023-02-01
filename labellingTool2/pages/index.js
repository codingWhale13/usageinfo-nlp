import { Heading, UnorderedList, ListItem } from "@chakra-ui/react";
import Link from "next/link";

function App() {
  return (
    <>
      <Heading as="h1">Hello world</Heading>
      <UnorderedList>
        <ListItem>
          <Link href="/local" passHref>
            Local Labelling
          </Link>
        </ListItem>
      </UnorderedList>
    </>
  );
}

export default App;
