import {  CustomUsageOptionFormTag, UsageOptionTag } from './UsageOptionTag';
import { Feature } from 'flagged';

const React = require('react');
const { Grid, GridItem, Heading, Tag, Divider, Wrap, Button, ButtonGroup, Stack, Center, Flex, Box, Spacer} =  require('@chakra-ui/react');
const {StarIcon, ArrowRightIcon, ArrowLeftIcon, RepeatClockIcon } = require('@chakra-ui/icons');

const {ReviewTokenAnnotator} = require('./ReviewTokenAnnotator');
const { Card } = require('./Elements');

export function Review(props){

        const { review } = props;

        const deleteAnnotation = (annotation) => {
            props.onSaveAnnotations(props.review.label.annotations.filter(annotationA => annotationA !== annotation));
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
            props.onSaveAnnotations([]);
            props.onSaveCustomUsageOptions([]);
        }
        if(!review){
            return 'No review';
        }
        return <Card>

        <Grid
        templateAreas={`"header category"
                        "main nav"`}
        gridTemplateRows={'60px 1fr'}
        gridTemplateColumns={'1fr 350px'}
        gap='3'
        fontSize={'lg'}
        >
        <GridItem pl='2' pt='1'  area={'header'} overflow={'hidden'}>
            <Card>
                <Heading as='h3' size='md' textAlign='left'>{review.product_title} </Heading>
            </Card>
                
            </GridItem>
            <GridItem pr='2' pt='2'  area={'category'} align={'right'}>
                <Tag size='lg'>{review.product_category}</Tag>
                <Divider m={2}/>
            </GridItem>

        <GridItem pl='2' pt='2' pr='2' area={'main'} borderRight='1px' borderColor='gray.100'>
            <ReviewTokenAnnotator 
                        review_body={review.review_body}
                        annotations={props.annotations}
                        onSaveAnnotations={props.onSaveAnnotations}
            />
        </GridItem>
        <GridItem pt='2' pl='2' area={'nav'}>
            <Center> 
                <ButtonGroup>
                    <Button onClick={() => {
                        props.navigateToPrevious();
                    }
                    } leftIcon={<ArrowLeftIcon/>} size='md'>
                        Previous
                    </Button>
                    <Feature name="localLabelling">
                        {props.isFlagged ? 
                            <Button colorScheme='red' onClick={() => props.onSaveFlag(false)} size='md'>
                                <StarIcon />
                                Remove flag
                            </Button>: 
                        <Button colorScheme='pink' onClick={() => props.onSaveFlag(true)} size='md'>
                            Flag for follow up
                        </Button>
                        }
                    </Feature>
                    <Button onClick={resetAnnotation} colorScheme='pink' leftIcon={<RepeatClockIcon/>} size='md'>
                        Reset
                    </Button>
                    
                    <Button type='submit' onClick={props.navigateToNext} rightIcon={<ArrowRightIcon/>} size='md'>
                        Next
                    </Button>
                </ButtonGroup>                    
            </Center>

            <Divider m={2}/>
            <Heading as='h5' size='sm' paddingY={2}>Selected usage options</Heading>
            <CustomUsageOptionFormTag 
                onSave={(newCustomUsageOption) => {
                    if(!props.review.label.customUsageOptions.includes(newCustomUsageOption)){
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

        
            
                {/* <Grid
                templateAreas={`
                                "nav main"
                                "nav footer"`}
                gridTemplateRows={'200px 600px'}
                gridTemplateColumns={'300px 800px'}
                gap='4'
            >
        
            <GridItem pl='2' area={'nav'}>
                <Heading as='h5' textAlign='left'>{review.product_title}</Heading>
                <Divider m={2}/>
                <Tag >{review.product_category}</Tag>
                <Divider m={2}/>
                <Heading as='h5' size='sm' paddingY={2}>Selected usage options</Heading>
                <Wrap spacing={2}>
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

                <CustomUsageOptionFormTag 
                    onSave={(newCustomUsageOption) => {
                        if(!props.review.label.customUsageOptions.includes(newCustomUsageOption)){
                            props.onSaveCustomUsageOptions(props.review.label.customUsageOptions.concat(newCustomUsageOption));
                        }
                    }}
                />
                </Wrap>
               
            </GridItem>
            <GridItem pl='2'  area={'main'}>
                <ButtonGroup>
                    <Button onClick={() => {
                        props.navigateToPrevious();
                    }
                    } leftIcon={<ArrowLeftIcon/>}>
                        Previous
                    </Button>
                    <Feature name="localLabelling">
                        {props.isFlagged ? 
                            <Button colorScheme='red' onClick={() => props.onSaveFlag(false)}>
                                <StarIcon />
                                Remove flag
                            </Button>: 
                        <Button colorScheme='pink' onClick={() => props.onSaveFlag(true)}>
                            Flag for follow up
                        </Button>
                        }
                    </Feature>
                    <Button onClick={resetAnnotation} colorScheme='pink' leftIcon={<RepeatClockIcon/>}>
                        Reset
                    </Button>
                    
                    <Button type='submit' onClick={props.navigateToNext} rightIcon={<ArrowRightIcon/>}>
                        Next
                    </Button>
                </ButtonGroup>

                <Divider m={2}/>

                <ReviewTokenAnnotator 
                    review_body={review.review_body}
                    annotations={props.annotations}
                    onSaveAnnotations={props.onSaveAnnotations}
                />
            </GridItem>
        </Grid> */}
      </Card>
}

