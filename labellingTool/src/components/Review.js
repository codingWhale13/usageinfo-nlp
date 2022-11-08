import { AnnotationUsageOptionTag, CustomUsageOptionFormTag, UsageOptionTag } from './UsageOptionTag';

const React = require('react');
const { Grid, GridItem, Heading, Tag, Divider, Wrap } =  require('@chakra-ui/react');
const {ReviewTokenAnnotator} = require('./ReviewTokenAnnotator');
const { Card } = require('./Elements');

export function Review(props){

        const { review } = props;
        if(!review){
            return 'No review';
        }
        return <Card>
            <Grid
                templateAreas={`
                                "nav main"
                                "nav footer"`}
                gridTemplateRows={'200px 600px'}
                gridTemplateColumns={'300px 800px'}
                gap='1'
            >
        
            <GridItem pl='2' area={'nav'}>
                <Heading as='h5' textAlign='left'>{review.product_title}</Heading>
                <Divider m={2}/>
                <Tag >{review.product_category}</Tag>
                <Divider m={2}/>
                <Heading as='h5' size='sm' padding={2}>Selected usage options</Heading>
                <Wrap spacing={2}>
                {props.review.label.annotations.map((annotation) => <AnnotationUsageOptionTag 
                    annotation={annotation}
                    onDelete = {() => {
                        props.onSaveAnnotations(props.review.label.annotations.filter(annotationA => annotationA !== annotation))
                    }}
                    />
                )}
                {props.review.label.customUsageOptions.map(customUsageOption => <UsageOptionTag
                    usageOption={customUsageOption}
                    onDelete={() => {
                        props.onSaveCustomUsageOptions(props.review.label.customUsageOptions.filter(usageOptionA => usageOptionA !== customUsageOption));
                    }} 
                >

                </UsageOptionTag>)}
                <CustomUsageOptionFormTag 
                    onSave={(newCustomUsageOption) => props.onSaveCustomUsageOptions(props.review.label.customUsageOptions.concat(newCustomUsageOption))}
                />
                </Wrap>
               
            </GridItem>
            <GridItem pl='2'  area={'main'}>
                <ReviewTokenAnnotator 
                    review_body={review.review_body}
                    isFlagged={props.isFlagged}
                    annotations={props.annotations}
                    onSaveAnnotations={props.onSaveAnnotations}
                    onSaveFlag={props.onSaveFlag}
                    navigateToNext={props.navigateToNext}
                    navigateToPrevious={props.navigateToPrevious}
                />
            </GridItem>
        </Grid>
      </Card>
}

