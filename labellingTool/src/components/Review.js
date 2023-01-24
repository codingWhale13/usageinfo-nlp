import { Feature } from 'flagged';
import {
  IS_FLAGGED,
  ANNOTATIONS,
  CUSTOM_USAGE_OPTIONS,
  IS_GOLDEN_DATASET,
  PREDICTED_USAGE_OPTIONS,
  PREDICTED_USAGE_OPTION_LABEL,
  PREDICTED_USAGE_OPTIONS_VOTE,
} from '../utils/labelKeys';
import { Button, Flex, ButtonGroup, Divider, Text, Grid, Tag, GridItem, VStack, Stack } from '@chakra-ui/react';
import { annotationsToUsageOptions } from '../utils/conversion';
import { AnnotationsEditor } from './Editors/AnnotationsEditor';
import { CustomUsageOptionsEditor } from './Editors/CustomUsageOptionsEditor';
import { UsageOptionsRatingEditor } from './Editors/UsageOptionsRatingEditor';
import { getFeatureFlags } from '../featureFlags';

const React = require('react');

const {
  StarIcon,
  ArrowRightIcon,
  ArrowLeftIcon,
  RepeatClockIcon,
  WarningIcon,
} = require('@chakra-ui/icons');

const { ReviewTokenEditor } = require('./Editors/ReviewTokenEditor');
const { Card } = require('./Elements');


export function Review(props) {
  const { review } = props;
  const { isPreviousDisabled, isNextDisabled } = props;
  const features = getFeatureFlags();
  const resetAnnotation = () => {
    props.saveLabel(ANNOTATIONS, []);
    props.saveLabel(CUSTOM_USAGE_OPTIONS, []);
    props.saveLabel(PREDICTED_USAGE_OPTIONS, review.label[PREDICTED_USAGE_OPTIONS].map(usageOption => { return { [PREDICTED_USAGE_OPTION_LABEL]: usageOption[PREDICTED_USAGE_OPTION_LABEL], [PREDICTED_USAGE_OPTIONS_VOTE]: NaN } }));
  };

  const saveCustomUsageOption = newCustomUsageOption => {
    if (
      !review.label.customUsageOptions.includes(newCustomUsageOption) &&
      !annotationsToUsageOptions(review.label[ANNOTATIONS]).includes(
        newCustomUsageOption
      ) &&
      newCustomUsageOption !== ''
    ) {
      props.saveLabel(
        CUSTOM_USAGE_OPTIONS,
        review.label.customUsageOptions.concat(newCustomUsageOption)
      );
    }
  };

  if (!review) {
    return 'No review';
  }

  const roundToTwoDecimals = num => {
    return Math.round(num * 100) / 100;
  };

  const inspectionTimeInSeconds = roundToTwoDecimals(
    review.workerInspectionTime / 1000
  );
  const wordsPerMinute = roundToTwoDecimals(
    review.review_body.split(' ').length / (inspectionTimeInSeconds / 60)
  );
  return (
    <Card>
      <Grid
        templateAreas={`"header nav"
                        "main nav"`}
        gridTemplateRows={'140px 1fr'}
        gridTemplateColumns={'1fr 350px'}
        gap="3"
        fontSize={'lg'}
      >
        <GridItem pl="0" pt="1" area={'header'}>
          <Card>
            <Text
              fontWeight={'bold'}
              fontSize={'20px'}
              as="h3"
              size="md"
              textAlign="left"
              noOfLines={1}
            >
              {review.product_title}{' '}
            </Text>
            <Divider m={2} />
            <VStack
              direction={['row', 'column']}
              alignItems="start"
            >
              <Feature name="localLabelling">
                <Stack direction={['column', 'row']}>
                  <Feature name="reviewLabelling">
                    <Tag size="lg">{review.review_id}</Tag>
                    <Tag size="lg">{review.workerId}</Tag>
                    <Tag size="lg">Time: {inspectionTimeInSeconds}s</Tag>
                    <Tag size="lg">Words per minute: {wordsPerMinute}</Tag>
                  </Feature>
                </Stack>
              </Feature>
              <Tag size="lg" colorScheme="blue">
                {review.product_category}
              </Tag>
            </VStack>
          </Card>
        </GridItem>

        <GridItem
          pl="2"
          pt="2"
          pr="2"
          area={'main'}
          borderRight="1px"
          borderColor="gray.100"
        >
          <ReviewTokenEditor
            saveLabel={props.saveLabel}
            review_body={review.review_body}
            annotations={review.label[ANNOTATIONS]}
            customUsageOptions={review.label[CUSTOM_USAGE_OPTIONS]}
          />
        </GridItem>
        <GridItem p="2" area={'nav'}>
          <Stack>
            <Feature name="localLabelling">
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
            </Feature>
          </Stack>

          <Flex
            alignItems={'center'}
            minWidth="max-content"
            direction="row"
            mt={2}
            gap="2"
          >
            <ButtonGroup gap="1">
              <Button
                onClick={() => {
                  props.navigateToPrevious();
                }}
                leftIcon={<ArrowLeftIcon />}
                size="md"
                colorScheme="gray"
                isDisabled={isPreviousDisabled}
              >
                Previous
              </Button>

              <Button
                onClick={resetAnnotation}
                colorScheme="pink"
                leftIcon={<RepeatClockIcon />}
                size="md"
                isDisabled={
                  review.label[ANNOTATIONS].length === 0 &&
                  review.label[CUSTOM_USAGE_OPTIONS].length === 0 &&
                  review.label[PREDICTED_USAGE_OPTIONS].length === 0
                }
              >
                Reset
              </Button>

              <Button
                type="submit"
                onClick={props.navigateToNext}
                rightIcon={<ArrowRightIcon />}
                size="md"
                isDisabled={isNextDisabled}
              >
                Next
              </Button>
            </ButtonGroup>
          </Flex>

          <Divider my={4} />
          {features.ratePredictedUseCases ?

            <UsageOptionsRatingEditor
              predictedUsageOptions={review.label[PREDICTED_USAGE_OPTIONS]}
              saveLabel={props.saveLabel}
            />
            :
            <>
              <AnnotationsEditor
                annotations={review.label[ANNOTATIONS]}
                saveLabel={props.saveLabel}
                saveCustomUsageOption={saveCustomUsageOption}
              />

              <Divider my={4} />
              <CustomUsageOptionsEditor
                customUsageOptions={review.label[CUSTOM_USAGE_OPTIONS]}
                annotations={review.label[ANNOTATIONS]}
                saveLabel={props.saveLabel}
                saveCustomUsageOption={saveCustomUsageOption}
              />
            </>


          }
        </GridItem>
      </Grid>
    </Card>
  );
}



