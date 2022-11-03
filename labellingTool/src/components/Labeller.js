import { Button, Progress, Heading, Container,   Stat,
    StatLabel,
    StatNumber,
    Flex, Spacer, ButtonGroup, Box } from '@chakra-ui/react'
import { Review } from './Review';
import { Card } from './Elements';
const React = require('react');
const {CSVUpload} = require('./CSVUpload');
const Papa = require('papaparse');
const { POSITIVE_TAG } = require('../utils/tags');
const _ = require('lodash');

export class Labeller extends React.Component{
    constructor(props){
        super(props);

        this.state = {
            reviews: [],
            reviewIndex : 0,
            maxReviewIndex: 0
        };
    }

    parseReviews = (e) => {
        Papa.parse(e.target.files[0], {
            header: true,
            skipEmptyLines: true,
            delimiter: '\t',
            complete: (csv) => {
                const reviews = csv.data;

                for (const review of reviews) {
                        review.label = {
                            isFlagged: false,
                            annotations: []
                        }
                }
                this.setState({reviews: reviews});
            
            },
            error : (error) => {
                console.error(error)
            }
        });
    }

    saveReviewFlag = (isFlagged, i) => {
        const reviews = [...this.state.reviews];
        reviews[i].label.isFlagged = isFlagged;
        this.setState({reviews: reviews});

    }
    saveReviewsAnnotations = (annotations, i) => {
        const reviews = [...this.state.reviews];
        reviews[i].label.annotations = annotations;
        this.setState({reviews: reviews});
    }

    exportLabelsToCSV = () => {
        function annotationsToLabel(annotations){
            if(annotations.length === 0){
                return '-';
            }
            return annotations.map((annotation) => [annotation.start, annotation.end, annotation.tag === POSITIVE_TAG ? 1 : 0]);
        }
        const reviews = _.cloneDeep(this.state.reviews);
        
        for(let i =0; i < reviews.length; i++){
            const review = reviews[i];
            review.is_flagged = review.label.isFlagged;
            if(i <= this.state.maxReviewIndex){
                review.label = annotationsToLabel(review.label.annotations);
            }
            else{
                review.label = null;
            }
        }
        const csv = Papa.unparse(reviews, {delimiter:'\t'});
        console.log('csv:',csv);
        var blob = new Blob([csv], { type: 'text/tsv;charset=utf-8;' });
        let encodedUri = URL.createObjectURL(blob);
        let link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "my_data2.tsv");
        document.body.appendChild(link); // Required for FF

        link.click();
    }

    

    render(){
        const reviewLabel = this.state.reviews.length ? this.state.reviews[this.state.reviewIndex].label : {};
        console.log(this.state, reviewLabel);
        return (<Container maxWidth='1300px'>

            {this.state.reviews.length === 0 &&
                <>
                    <h1>Bitte ein .tsv Datei mit Reviews hochladen</h1>
                    <CSVUpload 
                        onUpload={this.parseReviews}
                    />
                 </>
            }
           

            
            {this.state.reviewIndex < this.state.reviews.length &&
                <>  
                   <Card spacing={2} mb={2}>

                   <Flex minWidth='max-content' alignItems='center' gap='2'>
                    <Box>

                    
                   <Heading as='h2'>Label reviews</Heading>
                   <Stat>
                        <StatLabel>Labelled reviews</StatLabel>
                        <StatNumber>{this.state.reviewIndex + 1}/{this.state.reviews.length}</StatNumber>
                        </Stat>
                     </Box>
                    <Spacer />
                    <ButtonGroup gap='2'>
                        <Button colorScheme='teal' size='lg' onClick={this.exportLabelsToCSV}>
                                    Export
                        </Button>
                    </ButtonGroup>
                    </Flex>
                        
                        
              
            <Progress value={((this.state.reviewIndex + 1)/ this.state.reviews.length) * 100 } />
            </Card>

                    <Review  review={this.state.reviews[this.state.reviewIndex]}
                        onSaveAnnotations={(annotations) => {
                            this.saveReviewsAnnotations(annotations, this.state.reviewIndex)
                        }}
                        onSaveFlag={(isFlagged) => {
                            this.saveReviewFlag(isFlagged, this.state.reviewIndex);
                        }}
                        navigateToNext={() => {
                            this.setState({reviewIndex: this.state.reviewIndex + 1, maxReviewIndex: Math.max(this.state.reviewIndex + 1, this.state.maxReviewIndex)});
                        }}
                        navigateToPrevious={() => {
                            if(this.state.reviewIndex > 0){
                                this.setState({reviewIndex: this.state.reviewIndex - 1});
                            }
                        }}

                        isFlagged={reviewLabel ? reviewLabel.isFlagged : false}
                        annotations={reviewLabel ? reviewLabel.annotations : []}
                    />
                </>
            }
      </Container>);
    }


}