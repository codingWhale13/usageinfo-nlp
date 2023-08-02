import {
  Button,
  AlertDialog,
  AlertDialogBody,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogContent,
  AlertDialogOverlay,
  Heading,
  Text,
  Stack,
  UnorderedList,
  Image,
  ListItem,
} from '@chakra-ui/react';

import HowToImage from './how-to-high2.jpg';
import { useDisclosure } from '@chakra-ui/react';
import React from 'react';

export function InstructionsAlertDialog() {
  const { isOpen, onOpen, onClose } = useDisclosure({ defaultIsOpen: false });
  const cancelRef = React.useRef();

  return (
    <>
      <Button colorScheme="blue" onClick={onOpen} size="lg">
        Instructions
      </Button>

      <AlertDialog
        isOpen={isOpen}
        leastDestructiveRef={cancelRef}
        onClose={onClose}
        size="6xl"
      >
        <AlertDialogOverlay>
          <AlertDialogContent>
            <AlertDialogHeader fontSize="lg" fontWeight="bold">
              Instructions
            </AlertDialogHeader>

            <AlertDialogBody>
              <Stack spacing={4} pl="2">
                <Text>
                  The goal of this project is to extract usage information from
                  Amazon product reviews. For every review, your job is to
                  extract usage options that the review author describes. You can
                  either write a custom text or mark each usage option in the
                  review body directly.
                </Text>
                <Heading as="h4" size="md">
                  What counts as a usage option?
                </Heading>
                <Text>
                  A usage option is anything that answers the question "
                  <Text as="b">What can this product be used for/as?</Text>". We
                  are only interested in <Text as="b">positive</Text> usage
                  options. You should only include the necessary information to
                  understand the usage options and be as precise as possible.
                  Reviews may contain more than one usage option. In this case,
                  each usage option must be labeled, each containing all information
                  needed to make sense independently.
                </Text>
                <Text>
                  Examples of what should be labeled as a usage option:
                </Text>
                <UnorderedList stylePosition={'inside'}>
                  <ListItem>
                    "Great bag to travel in Europe over the day" -{'>'} "
                    <Text as="b">bag for European day travel</Text>"
                  </ListItem>
                  <ListItem>
                    "Not only can you <Text as="b">chop wood</Text>, you can{' '}
                    <Text as="b">break concrete</Text>,{' '}
                    <Text as="b">dig a fire pit</Text>, use it as a{' '}
                    <Text as="b">climbing aid in the woods</Text>, and{' '}
                    <Text as="b">fight</Text> with it as a last resort."
                  </ListItem>
                  <ListItem>
                    "great for blending fruits and vegetables" -{'>'} "
                    <Text as="b">blending fruits</Text>", "
                    <Text as="b">blending vegetables</Text>"
                  </ListItem>
                </UnorderedList>
                <Heading as="h4" size="md">
                  What does <Text as="b">not</Text> count as a usage option?
                </Heading>
                <UnorderedList stylePosition={'inside'}>
                  <ListItem>
                    References to similar products (
                    <Text as="s">replacement for GXSF733</Text>,{' '}
                    <Text as="s">better than my iPhone 8</Text>, ...)
                  </ListItem>
                  <ListItem>
                    Target audience (<Text as="s">babies</Text>,{' '}
                    <Text as="s">business people</Text>, ...)
                  </ListItem>
                  <ListItem>
                    Product attributes (<Text as="s">beautiful</Text>,{' '}
                    <Text as="s">robust</Text>, ...)
                  </ListItem>
                  <ListItem>
                    Gifts and presents (
                    <Text as="s">bought this as a birthday present</Text>,{' '}
                    <Text as="s">great gift</Text>, ...)
                  </ListItem>
                  <ListItem>
                    Negative use cases (
                    <Text as="s">
                      I hoped this would be good for playing tennis but it
                      wasn't!
                    </Text>
                    )
                  </ListItem>
                </UnorderedList>

                <Heading as="h3" size="md">
                  How to use the labeling tool
                </Heading>
                <Image src={HowToImage} />
              </Stack>
            </AlertDialogBody>

            <AlertDialogFooter>
              <Button colorScheme="green" onClick={onClose} ml={3}>
                I understand the instructions
              </Button>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialogOverlay>
      </AlertDialog>
    </>
  );
}
