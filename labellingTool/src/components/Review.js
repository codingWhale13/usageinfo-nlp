import { CustomUsageOptionFormTag, UsageOptionTag } from './UsageOptionTag';
import { Feature } from 'flagged';

const React = require('react');

const {
  Grid,
  GridItem,
  Heading,
  Tag,
  Divider,
  Wrap,
  Button,
  ButtonGroup,
  Stack,
  Flex,
  Text,
  Spacer,
  VStack
} = require('@chakra-ui/react');
const {
  StarIcon,
  ArrowRightIcon,
  ArrowLeftIcon,
  RepeatClockIcon,
  WarningIcon,
} = require('@chakra-ui/icons');

const { ReviewTokenAnnotator } = require('./ReviewTokenAnnotator');
const { Card } = require('./Elements');

export function Review(props) {
  const { review } = props;
  const { isPreviousDisabled, isNextDisabled } = props;
  
  const annotationsToUsageOptions = annotations => {
    return annotations.map(annotation => annotation.tokens.join(' ')).flat();
  };

  const uniqueAnnotations = (annotations) => {
    return annotations.filter(
        (annotation, index) => annotations.map(annotation => annotation.tokens.join(' ')).indexOf(annotation.tokens.join(' ')) === index
    );
    };

  const saveAnnotations = (annotations) => {
    const usageOptions = annotationsToUsageOptions(annotations);
    props.onSaveCustomUsageOptions(
      review.label.customUsageOptions.filter(
        usageOptionA => !usageOptions.includes(usageOptionA)
      )
    );
    props.onSaveAnnotations(annotations);
  };

  const deleteAnnotation = (annotation) => {
    props.onSaveAnnotations(
      review.label.annotations.filter(
        annotationA => annotationA.tokens.join(' ') !== annotation.tokens.join(' ')
      )
    );
  };
  

  const deleteCustomUsageOption = (customUsageOption) => {
    props.onSaveCustomUsageOptions(
      review.label.customUsageOptions.filter(
        usageOptionA => usageOptionA !== customUsageOption
      )
    );
  };

  const resetAnnotation = () => {
    props.onSaveAnnotations([]);
    props.onSaveCustomUsageOptions([]);
  };

  const saveCustomUsageOption =  (newCustomUsageOption) => {
        if (!review.label.customUsageOptions.includes(
            newCustomUsageOption
        ) &&
            !annotationsToUsageOptions(
                review.label.annotations
            ).includes(newCustomUsageOption) && newCustomUsageOption !== "") {
            props.onSaveCustomUsageOptions(
                review.label.customUsageOptions.concat(
                    newCustomUsageOption
                )
            );
        }
    };

    // const onUpdateAnnotationToCustomUsage = (annotation, newCustomUsageOption) => {
    //   const key = annotation.tokens.join(' ');
    //   console.log(review.label.updatedUsageOptions)
    //   let replacements = review.label.updatedUsageOptions.get(key);
    //   console.log(replacements)
    //   if (replacements === undefined) {
    //     replacements = [];
    //   }
    //   if (!replacements.includes(newCustomUsageOption)) {
    //     replacements.push(newCustomUsageOption);
    //   }
    //   review.label.updatedUsageOptions.set(key, replacements);
    //   props.onUpdateUsageOption(review.label.updatedUsageOptions);
    // };




    const updateAnnotation = (annotation) => {
      return (newCustomUsageOption) => {
        if (newCustomUsageOption === "") {
          deleteAnnotation(annotation);
        } else {
          saveCustomUsageOption(newCustomUsageOption);
          // onUpdateAnnotationToCustomUsage(annotation, newCustomUsageOption);
          deleteAnnotation(annotation);
        }
      };
    }

    const updateCustomUsageOption = (customUsageOption) => {
        return updatedCustomUsageOption => {
            if (annotationsToUsageOptions(review.label.annotations).includes(updatedCustomUsageOption) || updatedCustomUsageOption === "") {
                deleteCustomUsageOption(customUsageOption);
            } else {
                props.onSaveCustomUsageOptions(
                    review.label.customUsageOptions.map(
                        usageOptionA => usageOptionA === customUsageOption ? updatedCustomUsageOption : usageOptionA
                    ).filter((usageOptionA, index, self) => self.indexOf(usageOptionA) === index)
                ); // different from saveCustomUsageOption because we do not append to list
            }
        };
    }

  if (!review) {
    return 'No review';
  }

  const roundToTwoDecimals = (num) => {
    return Math.round(num * 100) / 100;
  }

  const inspectionTimeInSeconds = roundToTwoDecimals(review.workerInspectionTime/1000);
  const wordsPerMinute = roundToTwoDecimals(review.review_body.split(' ').length / (inspectionTimeInSeconds /60));
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
            <VStack direction={[ 'row', 'column']} style={{'align-items': 'start'}}> 
              <Feature name="localLabelling">
              <Stack direction={['column', 'row']}>
                <Tag size="lg">{review.review_id}</Tag>
                <Tag size="lg">{review.workerId}</Tag>
                <Tag size="lg">Time: {inspectionTimeInSeconds}s</Tag>
                <Tag size="lg">Words per minute: {wordsPerMinute}</Tag>
                </Stack>

              </Feature>
              <Tag size="lg" colorScheme='blue'>{review.product_category}</Tag>
              

            
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
          <ReviewTokenAnnotator
            review_body={review.review_body}
            annotations={review.label.annotations}
            onSaveAnnotations={saveAnnotations}
          />
        </GridItem>
        <GridItem pt="2" pl="2" area={'nav'}>
          <Stack>
            <Feature name="localLabelling">

      
            <ButtonGroup gap="2">
              {!review.label.isLabelGood ?
              <>
                <Button
                onClick={resetAnnotation}
                colorScheme="red"
                size="md"
              >
                Mark label as bad
              </Button>                

              <Button
                type="submit"
                onClick={props.navigateToNext}
                size="md"
                colorScheme="green"
              >
                Mark label as good
              </Button>
              </>
              :
              <>
               <Button
                onClick={resetAnnotation}
                colorScheme="red"
                size="md"
              >
                Mark label bad
              </Button>                

              <Button
                type="submit"
                onClick={props.navigateToNext}
                rightIcon={<ArrowRightIcon />}
                size="md"
                colorScheme="green"
              >
                Mark label good
              </Button>
              </>
              }
            </ButtonGroup>
              {review.label.isFlagged ? (
                               <Button
                  leftIcon={<StarIcon />}
                  colorScheme="red"
                  onClick={() => props.onSaveFlag(false)}
                  size="md"
                >
                  Remove flag
                </Button>
              ) : (
                <Button
                  leftIcon={<WarningIcon />}
                  colorScheme="red"
                  onClick={() => props.onSaveFlag(true)}
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
            <ButtonGroup gap="2">
              <Button
                onClick={() => {
                  props.navigateToPrevious();
                }}
                leftIcon={<ArrowLeftIcon />}
                size="md"
                colorScheme="gray"
                disabled={isPreviousDisabled}
              >
                Previous
              </Button>

              <Button
                onClick={resetAnnotation}
                colorScheme="pink"
                leftIcon={<RepeatClockIcon />}
                size="md"
                disabled={
                  review.label.annotations.length === 0 && review.label.customUsageOptions.length === 0
                }
              >
                Reset
              </Button>                
        
              <Button
                type="submit"
                onClick={props.navigateToNext}
                rightIcon={<ArrowRightIcon />}
                size="md"
                disabled={isNextDisabled}
              >
                Next
              </Button>
            </ButtonGroup>
          </Flex>

          <Divider my={4} />
          <Heading as="h5" size="sm" paddingY={2}>
            Annotated usage options
          </Heading>
          <Wrap spacing={2} pt="2">
            
            {uniqueAnnotations(review.label.annotations).map(annotation => (
              <UsageOptionTag
                usageOption={annotation.tokens.join(' ')}
                key={annotation.tokens.join(' ')}
                onDeleteUsageOption={() => deleteAnnotation(annotation)}
                onUpdateUsageOption={updateAnnotation(annotation)}
              ></UsageOptionTag>
            ))}

            </Wrap>
            <Divider my={4} />

            <Heading as="h5" size="sm" paddingY={2}>
                Custom usage options
            </Heading>

            
          <CustomUsageOptionFormTag
            onSave={saveCustomUsageOption}
          />
            <Wrap spacing={2} pt="5">
            {review.label.customUsageOptions.map(customUsageOption => (
              <UsageOptionTag
                usageOption={customUsageOption}
                key={customUsageOption}
                onDeleteUsageOption={() => deleteCustomUsageOption(customUsageOption)}
                onUpdateUsageOption={updateCustomUsageOption(customUsageOption)}
              ></UsageOptionTag>
            ))}
            </Wrap>
        </GridItem>
      </Grid>
    </Card>
  );

  
}
