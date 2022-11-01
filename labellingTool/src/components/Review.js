const React = require('react');
const { Grid, GridItem, Heading, Tag, Divider } =  require('@chakra-ui/react');
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
                <Tag>{review.product_category}</Tag>
            </GridItem>
            <GridItem pl='2'  area={'main'}>
                <ReviewTokenAnnotator 
                    review_body={review.review_body}
                    onSave={props.onSave}
                />
            </GridItem>
        </Grid>
      </Card>
}

