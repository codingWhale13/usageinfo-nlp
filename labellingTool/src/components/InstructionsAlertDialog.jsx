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
  OrderedList,
  ListItem,
} from '@chakra-ui/react';

import { useDisclosure } from '@chakra-ui/react';
import React from 'react';

export function InstructionsAlertDialog() {
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
        size="4xl"
      >
        <AlertDialogOverlay>
          <AlertDialogContent>
            <AlertDialogHeader fontSize="lg" fontWeight="bold">
              Instructions
            </AlertDialogHeader>

            <AlertDialogBody>
              <Stack spacing={4}>
                <Text>
                  The goal of this project is to extract usage information from
                  Amazon product reviews. For every review, your job is to
                  extract usage options that the review author describes. If
                  possible, mark a usage option directly in the review. If
                  necessary, you can also write custom text to describe a usage
                  option.
                </Text>
                <Heading as="h4" size="md">
                  What counts as a usage option?
                </Heading>
                <Text>
                  A usage option is anything that answers the question "
                  <Text as="b">What can this product be used for/as?</Text>". We
                  are only interested in <Text as="b">positive</Text> usage
                  options. You must label the{' '}
                  <Text as="b">shortest phrase possible</Text> that includes the
                  full usage option. Reviews may contain more than one usage
                  option. In this case, each usage option must be labeled, each
                  containing all information needed to make sense independently.
                </Text>
                <Text>
                  Examples of what should be labeled as a usage option:
                </Text>
                <Text>
                  <UnorderedList>
                    <ListItem>
                      {' '}
                      "A wonderful tool for{' '}
                      <Text as="b">cleaning the bathroom</Text>!" (marked
                      directly in review){' '}
                    </ListItem>
                    <ListItem>
                      "Not only can you <Text as="b">chop wood</Text>, you can{' '}
                      <Text as="b">break concrete</Text>,{' '}
                      <Text as="b">dig a fire pit</Text>, use it as a{' '}
                      <Text as="b">climbing aid in the woods</Text>, and{' '}
                      <Text as="b">fight</Text> with it as a last resort."
                      (marked directly in review)
                    </ListItem>
                    <ListItem>
                      "great for blending fruits and vegetables" -{'>'} "
                      <Text as="b">blending fruits</Text>", "
                      <Text as="b">blending vegetables</Text>" (free text)
                    </ListItem>
                  </UnorderedList>
                </Text>
                <Heading as="h4" size="md">
                  What does <Text as="b">not</Text> count as a usage option?
                </Heading>
                <Text>
                  <UnorderedList>
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
                </Text>

                <Heading as="h3" size="md">
                  How to use the labeling tool
                </Heading>
                <Text>
                  <OrderedList>
                    <ListItem>
                      In one review, there can be any number (including 0) of
                      usage options.
                    </ListItem>
                    <ListItem>
                      Usages can be either
                      <UnorderedList>
                        <ListItem>
                          marked and used exactly as it's written in the review
                          or
                        </ListItem>
                        <ListItem>
                          {' '}
                          written as free text; this allows:
                          <UnorderedList>
                            <ListItem>
                              avoid filler words used by review author
                            </ListItem>
                            <ListItem> correct misspellings</ListItem>
                            <ListItem>
                              {' '}
                              articulate something that's clear from context but
                              not written explicitly
                            </ListItem>
                          </UnorderedList>
                        </ListItem>
                      </UnorderedList>
                    </ListItem>
                  </OrderedList>
                </Text>
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
