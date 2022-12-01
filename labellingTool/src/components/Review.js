import {  CustomUsageOptionFormTag, UsageOptionTag } from './UsageOptionTag';
import { Feature } from 'flagged';

const React = require('react');
const { Grid, GridItem, Heading, Tag, Divider, Wrap, Button, ButtonGroup, Stack, Center, Flex, Box, Spacer, Text} =  require('@chakra-ui/react');
const {StarIcon, ArrowRightIcon, ArrowLeftIcon, RepeatClockIcon, WarningIcon } = require('@chakra-ui/icons');

const {ReviewTokenAnnotator} = require('./ReviewTokenAnnotator');
const { Card } = require('./Elements');

export function Review(props){

        const { review } = props;

        const annotationsToUsageOptions = (annotations) => {
            return annotations.map( (annotation) => annotation.tokens.join(' ')).flat();
        }

        const saveUniqueAnnonations = (annotations) => {
            const usageOptions = annotationsToUsageOptions(annotations);
            props.onSaveCustomUsageOptions(props.review.label.customUsageOptions.filter(usageOptionA => !usageOptions.includes(usageOptionA)));
            props.onSaveAnnotations(annotations);
        };

        const deleteAnnotation = (annotation) => {
            saveUniqueAnnonations(props.review.label.annotations.filter(annotationA => annotationA !== annotation));
        };

        const deleteCustomUsageOption = (customUsageOption) => {
            props.onSaveCustomUsageOptions(props.review.label.customUsageOptions.filter(usageOptionA => usageOptionA !== customUsageOption));
        };

        const deleteReplacementClassesMapping = (usageOption) => {
            const newReplacementClasses = new Map(props.review.label.replacementClasses);
            newReplacementClasses.delete(usageOption);
            props.onSaveReplacementClasses(newReplacementClasses);
        };

        const resetAnnotation = () => {
            saveUniqueAnnonations([]);
            props.onSaveCustomUsageOptions([]);
        }
        if(!review){
            return 'No review';
        }
        return <Card>

        <Grid
        templateAreas={`"header nav"
                        "main nav"`}
        gridTemplateRows={'110px 1fr'}
        gridTemplateColumns={'1fr 350px'}
        gap='3'
        fontSize={'lg'}
        >
        <GridItem pl='0' pt='1' area={'header'}>
            <Card>
            <Text fontWeight={'bold'} fontSize={'20px'}  as='h3' size='md' textAlign='left' noOfLines={1}>{review.product_title} </Text>    
            <Divider m={2}/>
            <Tag size='lg'>{review.product_category}</Tag>
            </Card>
            
            </GridItem>

        <GridItem pl='2' pt='2' pr='2' area={'main'} borderRight='1px' borderColor='gray.100'>
            <ReviewTokenAnnotator 
                        review_body={review.review_body}
                        annotations={review.label.annotations}
                        onSaveAnnotations={saveUniqueAnnonations}
            />
        </GridItem>
        <GridItem pt='2' pl='2' area={'nav'}>
        <Stack > <Feature name="localLabelling">
                        {props.isFlagged ? 
                            <Button leftIcon={<StarIcon />} colorScheme='red' onClick={() => props.onSaveFlag(false)} size='md'>
                                
                                Remove flag
                            </Button>: 
                        <Button leftIcon={<WarningIcon />} colorScheme='red' onClick={() => props.onSaveFlag(true)} size='md'>
                            
                            Flag for follow up
                        </Button>
                        }
            </Feature></Stack>
        
            <Flex  alignItems={'center'} minWidth='max-content'  direction='row' mt={2} gap='2'> 
            <ButtonGroup gap='2'>
            <Button onClick={() => {
                        props.navigateToPrevious();
                    }
                    } leftIcon={<ArrowLeftIcon/>} size='md' colorScheme='gray'>
                        Previous
                    </Button>
                    
                    <Button onClick={resetAnnotation} colorScheme='pink' leftIcon={<RepeatClockIcon/>} size='md'>
                        Reset
                    </Button>
                    
                    <Button type='submit' onClick={props.navigateToNext} rightIcon={<ArrowRightIcon/>} size='md'>
                        Next
                    </Button>        
            </ButtonGroup>
                    
            </Flex>

            <Divider my={4}/>
            <Heading as='h5' size='sm' paddingY={2}>Selected usage options</Heading>
            <CustomUsageOptionFormTag 
                onSave={(newCustomUsageOption) => {
                    if(!props.review.label.customUsageOptions.includes(newCustomUsageOption) && !annotationsToUsageOptions(props.review.label.annotations).includes(newCustomUsageOption)){
                        props.onSaveCustomUsageOptions(props.review.label.customUsageOptions.concat(newCustomUsageOption));
                    }
                }}
            />
            
            <Wrap spacing={2} pt='5'>
                {props.review.label.annotations.map((annotation) => 
                    <UsageOptionTag
                        annotation={annotation}
                        deleteAnnotation={deleteAnnotation}
                        replacementClasses={review.label.replacementClasses}
                        deleteReplacementClassesMapping={deleteReplacementClassesMapping}>
                    </UsageOptionTag>
                )}
                

                {props.review.label.customUsageOptions.map(customUsageOption => 
                <UsageOptionTag 
                    customUsageOption={customUsageOption}
                    replacementClasses={review.label.replacementClasses}
                    deleteCustomUsageOption={deleteCustomUsageOption}
                    deleteReplacementClassesMapping={deleteReplacementClassesMapping}
                >
                </UsageOptionTag>
                )}

                
                </Wrap>
        </GridItem>
        </Grid>
      </Card>
}

