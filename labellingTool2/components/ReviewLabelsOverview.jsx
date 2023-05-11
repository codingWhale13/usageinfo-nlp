import {
  Button,
  Divider,
  Text,
  Grid,
  Tag,
  GridItem,
  VStack,
  Stack,
  StackItem,
  HStack,
  Heading,
  Popover,
  PopoverTrigger,
  PopoverContent,
} from "@chakra-ui/react";
import { FaInfo } from "react-icons/fa";
import { ReviewTokenEditor } from "./Editors/ReviewTokenEditor";
const React = require("react");
const { ReviewHeader } = require("./ReviewHeader");

const { CustomCard } = require("./Elements");

export function ReviewLabelsOverview({ review, selectedLabelIds }) {
  return (
    <CustomCard>
      <Grid
        templateAreas={`"header nav"
                        "main nav"`}
        gridTemplateRows={"100px 1fr"}
        gridTemplateColumns={"700px 1fr"}
        gap="3"
        fontSize={"lg"}
      >
        <GridItem pl="0" pt="1" area={"header"}>
          <ReviewHeader review={review} />
        </GridItem>

        <GridItem
          pl="2"
          pt="2"
          pr="2"
          area={"main"}
          borderRight="1px"
          borderColor="gray.100"
        >
          <Text as="h5" fontWeight={"bold"} size="sm" textAlign="left">
            {review.review_headline}
          </Text>
          <ReviewTokenEditor
            review_body={review.review_body}
            isDisabled={true}
          />
        </GridItem>
        <GridItem p="2" area={"nav"}>
          <VStack spacing="24px" align="left">
            {Object.entries(review.labels).map(([labelId, label]) => (
              <>
                {selectedLabelIds.includes(labelId) && (
                  <>
                    <HStack spacing="10px">
                      <Heading>{labelId}</Heading>

                      <Popover trigger="hover">
                        <PopoverTrigger>
                          <Button size="sm">
                            <FaInfo />
                          </Button>
                        </PopoverTrigger>
                        <PopoverContent>
                          METADATA:
                          <pre>{JSON.stringify(label.metadata, null, 2)}</pre>
                          SCORES:
                          <pre>{JSON.stringify(label.scores, null, 2)}</pre>
                        </PopoverContent>
                      </Popover>
                    </HStack>

                    <Stack direction={["column", "row"]} wrap="wrap">
                      {label.usageOptions.length === 0 ? (
                        <StackItem margin={"0.25rem !important"}>
                          <Tag colorScheme="red" size="lg">
                            No usage options
                          </Tag>
                        </StackItem>
                      ) : (
                        label.usageOptions.map((usageOption) => (
                          <StackItem margin={"0.25rem !important"}>
                            <Tag size="lg">{usageOption}</Tag>
                          </StackItem>
                        ))
                      )}
                    </Stack>

                    <Divider />
                  </>
                )}
              </>
            ))}
          </VStack>
        </GridItem>
      </Grid>
    </CustomCard>
  );
}
