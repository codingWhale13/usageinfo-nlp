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
  ListItem,
} from '@chakra-ui/react';

import { useDisclosure } from '@chakra-ui/react';
import React from 'react';

export function InstructionsAlertDialogCuration() {
  const { isOpen, onOpen, onClose } = useDisclosure({ defaultIsOpen: true });
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
                  The goal of this project is to{' '}
                  <Text as="b">evaluate extracted usage options</Text> from
                  Amazon product reviews. For every review, your job is to
                  evaluate every usage option that was labelled in the review.
                  Your evaluation consists of two steps:
                </Text>
                <UnorderedList stylePosition={'inside'}>
                  <ListItem>
                    <Text as="b">Down- or Upvote</Text> each usage option
                    according to the rules below
                  </ListItem>
                  <ListItem>
                    <Text as="b">Down- or Upvote</Text> wether usage options
                    that are mentioned in the review are missing
                  </ListItem>
                </UnorderedList>
                <Heading as="h4" size="md">
                  What counts as a usage option?
                </Heading>
                <Text>
                  A usage option is anything that answers the question "
                  <Text as="b">What can this product be used for/as?</Text>". We
                  are only interested in <Text as="b">positive</Text> usage
                  options.
                </Text>
                <Text>
                  Examples of what should be labeled as a usage option:
                </Text>
                <UnorderedList stylePosition={'inside'}>
                  <ListItem>
                    {' '}
                    "A wonderful tool for{' '}
                    <Text as="b">cleaning the bathroom</Text>!"
                  </ListItem>
                  <ListItem>
                    "You can <Text as="b">chop wood</Text> and{' '}
                    <Text as="b">break concrete</Text> with it."
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
                  <ListItem>
                    Slang (
                    <Text as="s">
                      When I slammed a power chord on this electric guitar, my
                      hair blew back!
                    </Text>
                    )
                  </ListItem>
                </UnorderedList>
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
