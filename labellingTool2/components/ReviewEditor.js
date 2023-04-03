import { Feature } from "flagged";
import {
  IS_FLAGGED,
  ANNOTATIONS,
  CUSTOM_USAGE_OPTIONS,
  IS_GOLDEN_DATASET,
  PREDICTED_USAGE_OPTIONS,
  PREDICTED_USAGE_OPTION_LABEL,
  PREDICTED_USAGE_OPTIONS_VOTE,
  CONTAINS_MORE_USAGE_OPTIONS,
  IS_LABELLED,
} from "../utils/labelKeys";
import { GOOD_VOTE, BAD_VOTE } from "../utils/voteDefinitions";
import {
  Button,
  ButtonGroup,
  Divider,
  Text,
  Grid,
  GridItem,
  Stack,
  Card,
  CardBody,
  CardFooter,
  Heading,
} from "@chakra-ui/react";
import { annotationsToUsageOptions } from "../utils/conversion";
import { AnnotationsEditor } from "./Editors/AnnotationsEditor";
import { CustomUsageOptionsEditor } from "./Editors/CustomUsageOptionsEditor";
import { UsageOptionsRatingEditor } from "./Editors/UsageOptionsRatingEditor";
import { getFeatureFlags } from "../featureFlags";
import { ToggleButton } from "./Editors/UsageOptionsRatingEditor";
import { FaThumbsDown, FaThumbsUp, FaCheck } from "react-icons/fa";
import { ReviewHeader } from "./ReviewHeader";
const React = require("react");

const {
  StarIcon,

  RepeatClockIcon,
  WarningIcon,
} = require("@chakra-ui/icons");

const { ReviewTokenEditor } = require("./Editors/ReviewTokenEditor");
const { CustomCard } = require("./Elements");

export function ReviewEditor({ isDisabled = true, editLabelId, ...props }) {
  const { review } = props;
  const features = getFeatureFlags();
  const labelId = null;

  const resetAnnotation = () => {
    if (features.ratePredictedUseCases) {
      props.saveLabel(
        PREDICTED_USAGE_OPTIONS,
        review.label[PREDICTED_USAGE_OPTIONS].map((usageOption) => {
          return {
            [PREDICTED_USAGE_OPTION_LABEL]:
              usageOption[PREDICTED_USAGE_OPTION_LABEL],
            [PREDICTED_USAGE_OPTIONS_VOTE]: NaN,
          };
        })
      );
    } else {
      props.saveLabel(ANNOTATIONS, []);
      props.saveLabel(CUSTOM_USAGE_OPTIONS, []);
      props.saveLabel(IS_LABELLED, false);
    }
  };

  const saveCustomUsageOption = (newCustomUsageOption) => {
    if (
      !review.label.customUsageOptions.includes(newCustomUsageOption) &&
      !annotationsToUsageOptions(review.label[ANNOTATIONS]).includes(
        newCustomUsageOption
      ) &&
      newCustomUsageOption !== ""
    ) {
      props.saveLabel(
        CUSTOM_USAGE_OPTIONS,
        review.label.customUsageOptions.concat(newCustomUsageOption)
      );
      props.saveLabel(IS_LABELLED, true);
    }
  };

  if (!review) {
    return "No review";
  }

  return (
    <CustomCard>
      <Grid
        templateAreas={`"header nav"
                        "main nav"`}
        gridTemplateRows={"100px 1fr"}
        gridTemplateColumns={"1fr 350px"}
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
            saveLabel={props.saveLabel}
            review_body={review.review_body}
            annotations={review.label[ANNOTATIONS]}
            customUsageOptions={review.label[CUSTOM_USAGE_OPTIONS]}
            isDisabled={false}
          />
        </GridItem>
        <GridItem p="2" area={"nav"}>
          <Stack>
            <Feature name="localLabelling">
              <Card>
                <CardBody>
                  <Text align="center">Your label id: {editLabelId}</Text>
                </CardBody>
              </Card>
              {review.label[IS_GOLDEN_DATASET] ? (
                <Button
                  leftIcon={<StarIcon />}
                  colorScheme="yellow"
                  onClick={() => props.saveLabel(IS_GOLDEN_DATASET, false)}
                  size="md"
                >
                  Remove flag
                </Button>
              ) : (
                <Button
                  leftIcon={<WarningIcon />}
                  colorScheme="yellow"
                  onClick={() => props.saveLabel(IS_GOLDEN_DATASET, true)}
                  size="md"
                >
                  Flag for golden dataset
                </Button>
              )}
              {review.label[IS_FLAGGED] ? (
                <Button
                  leftIcon={<StarIcon />}
                  colorScheme="red"
                  onClick={() => props.saveLabel(IS_FLAGGED, false)}
                  size="md"
                >
                  Remove flag
                </Button>
              ) : (
                <Button
                  leftIcon={<WarningIcon />}
                  colorScheme="red"
                  onClick={() => props.saveLabel(IS_FLAGGED, true)}
                  size="md"
                >
                  Flag for follow up
                </Button>
              )}
              <Button
                onClick={resetAnnotation}
                colorScheme="pink"
                leftIcon={<RepeatClockIcon />}
                size="md"
                isDisabled={
                  (review.label[ANNOTATIONS].length === 0 &&
                    review.label[CUSTOM_USAGE_OPTIONS].length === 0 &&
                    !features.ratePredictedUseCases &&
                    !review.label[IS_LABELLED]) ||
                  (review.label[PREDICTED_USAGE_OPTIONS] &&
                    review.label[PREDICTED_USAGE_OPTIONS].length === 0 &&
                    features.ratePredictedUseCases) ||
                  isDisabled
                }
              >
                Reset
              </Button>
              <ToggleButton
                text={"I labelled this!"}
                isOn={review.label[IS_LABELLED]}
                onColor={"green"}
                leftIcon={<FaCheck />}
                onClick={(e) => {
                  props.saveLabel(IS_LABELLED, !review.label[IS_LABELLED]);
                  props.saveLabel(ANNOTATIONS, []);
                  props.saveLabel(CUSTOM_USAGE_OPTIONS, []);
                }}
              />
            </Feature>
          </Stack>

          <Divider my={4} />

          <Feature name="ratePredictedUseCases">
            <Heading as="h5" size="sm" paddingY={2}>
              Evaluate completeness of usage options
            </Heading>
            <Stack>
              <Card
                maxW="100%"
                variant="outline"
                sx={{ "--card-padding": "0.5rem" }}
              >
                <CardBody>
                  <Text>
                    All usage options mentioned in the review are listed
                  </Text>
                </CardBody>

                <CardFooter
                  justify="space-between"
                  flexWrap="wrap"
                  sx={{
                    "& > button": {
                      minW: "136px",
                    },
                  }}
                >
                  <ButtonGroup
                    direction="row"
                    spacing={3}
                    align="center"
                    size={"md"}
                  >
                    <ToggleButton
                      text={"Upvote"}
                      isOn={
                        review.label[CONTAINS_MORE_USAGE_OPTIONS] === BAD_VOTE
                      }
                      onColor={"green"}
                      leftIcon={<FaThumbsUp />}
                      onClick={(e) =>
                        props.saveLabel(CONTAINS_MORE_USAGE_OPTIONS, BAD_VOTE)
                      }
                    />

                    <ToggleButton
                      text={"Downvote"}
                      isOn={
                        review.label[CONTAINS_MORE_USAGE_OPTIONS] === GOOD_VOTE
                      }
                      onColor={"red"}
                      leftIcon={<FaThumbsDown />}
                      onClick={(e) =>
                        props.saveLabel(CONTAINS_MORE_USAGE_OPTIONS, GOOD_VOTE)
                      }
                    />
                  </ButtonGroup>
                </CardFooter>
              </Card>
            </Stack>

            <Divider my={4} />
          </Feature>

          {features.ratePredictedUseCases ? (
            <UsageOptionsRatingEditor
              predictedUsageOptions={review.label[PREDICTED_USAGE_OPTIONS]}
              saveLabel={props.saveLabel}
            />
          ) : (
            <>
              <Heading>{labelId}</Heading>
              <AnnotationsEditor
                annotations={review.label[ANNOTATIONS]}
                saveLabel={props.saveLabel}
                saveCustomUsageOption={saveCustomUsageOption}
                isDisabled={isDisabled}
              />

              <Divider my={4} />
              <CustomUsageOptionsEditor
                customUsageOptions={review.label[CUSTOM_USAGE_OPTIONS]}
                annotations={review.label[ANNOTATIONS]}
                saveLabel={props.saveLabel}
                saveCustomUsageOption={saveCustomUsageOption}
                isDisabled={isDisabled}
              />
            </>
          )}
        </GridItem>
      </Grid>
    </CustomCard>
  );
}
