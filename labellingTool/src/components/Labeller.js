import { Button, Progress, Heading, Container,   Stat,
    StatNumber,
    Spacer, ButtonGroup, Box, Flex, Center, Grid, GridItem} from '@chakra-ui/react'
import { Review } from './Review';
import { Card } from './Elements';
import { Feature } from 'flagged';

const React = require('react');
const {CSVUpload} = require('./CSVUpload');
const {JSONUpload} = require('./JSONUpload');

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
                            annotations: [],
                            customUsageOptions: [],
                            replacementClasses: new Map()
                        }
                }
                this.setState({reviews: reviews});            
            },
            error : (error) => {
                console.error(error)
            }
        });
    }

    parseJSONReviews = async (e) => {
        const file = e.target.files[0];
        const jsonData = JSON.parse(await file.text());
        this.setState({
            reviews: jsonData.reviews,
            reviewIndex: jsonData.maxReviewIndex,
            maxReviewIndex: jsonData.maxReviewIndex
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

    saveCustomUsageOptions = (customUsageOptions, i) => {
        const reviews = [...this.state.reviews];
        reviews[i].label.customUsageOptions = customUsageOptions;
        this.setState({reviews: reviews});
    }

    saveReplacementClasses = (replacementClasses, i) => {
        const reviews = [...this.state.reviews];
        reviews[i].label.replacementClasses = new Map(replacementClasses);
        this.setState({reviews: reviews});
    }

    exportReviewsToJSON = () => {
        const reviewState = {
            reviews: this.state.reviews,
            maxReviewIndex: this.state.maxReviewIndex
        };
        const jsonBlob = new Blob([JSON.stringify(reviewState)], { type: "text/plain;charset=utf-8" });
        this.downloadBlob(jsonBlob, 'my_data.json');

    }

    downloadBlob = (blob, fileName) => {
        const encodedUri = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", fileName);
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }   

    render(){
        const reviewLabel = (this.state.reviews.length && this.state.reviewIndex < this.state.reviews.length)  ? this.state.reviews[this.state.reviewIndex].label : {};
        console.log(this.state, reviewLabel);

    
        const exportButton = <>
            <Button colorScheme='teal' size='lg' onClick={this.exportReviewsToJSON}>
                Export to JSON
            </Button>
        </>;

        return (<Container maxWidth='1300px'>
            {this.state.reviews.length === 0 &&
                <Center h='100%'>
                    <Grid templateColumns='repeat(2, 1fr)'>
                        <GridItem>
                            <h1>Please upload a .tsv file:</h1>
                            <CSVUpload 
                                onUpload={this.parseReviews}
                            />
                        </GridItem>
                        <GridItem>
                            <h1>Or upload an already labelled .json file:</h1>
                            <JSONUpload
                                onUpload={this.parseJSONReviews}
                            />
                        </GridItem>
                    </Grid>
                    
                    
                </Center>
            }
           {this.state.reviewIndex < this.state.reviews.length &&
                <>
                   <Card spacing={2} mb={2}>
                        <Flex>
                        
                            <Box>
                                <Heading as='h5' size='md'>Label reviews</Heading>
                                <Stat>
                                    <StatNumber>{this.state.reviewIndex + 1}/{this.state.reviews.length}</StatNumber>
                                </Stat>
                            </Box>
                            <Spacer />
                            <Feature name="localLabelling">
                                <ButtonGroup gap='2'>
                                {exportButton}
                                </ButtonGroup>
                            </Feature>
                            
                        </Flex>

                        <Progress mt={1} value={((this.state.reviewIndex + 1) / this.state.reviews.length) * 100}  />
                    </Card>
                    
                    <Review review={this.state.reviews[this.state.reviewIndex]}
                        onSaveAnnotations={(annotations) => {
                            this.saveReviewsAnnotations(annotations, this.state.reviewIndex);
                        } }


                        onSaveCustomUsageOptions={(customUsageOptions) => {
                            this.saveCustomUsageOptions(customUsageOptions, this.state.reviewIndex);
                        }} 

                        onSaveFlag={(isFlagged) => {
                            this.saveReviewFlag(isFlagged, this.state.reviewIndex);
                        } }

                        onSaveReplacementClasses={(replacementClasses) => {
                            this.saveReplacementClasses(replacementClasses, this.state.reviewIndex);
                        }}

                        navigateToNext={() => {
                            this.setState({ reviewIndex: this.state.reviewIndex + 1, maxReviewIndex: Math.max(this.state.reviewIndex + 1, this.state.maxReviewIndex) });
                        } }
                        navigateToPrevious={() => {
                            if (this.state.reviewIndex > 0) {
                                this.setState({ reviewIndex: this.state.reviewIndex - 1 });
                            }
                        } }

                        isFlagged={reviewLabel ? reviewLabel.isFlagged : false} />
                </>
            }
            {(this.state.reviewIndex !== 0 && this.state.reviewIndex >= this.state.reviews.length) &&

            <ButtonGroup>
                <Button onClick={() => { this.setState({ reviewIndex: this.state.reviewIndex - 1 });}}>
                    Previous
                </Button>
                
                <Feature name="localLabelling">
                    {exportButton}
                </Feature>
            </ButtonGroup>
            }

      </Container>);
    }


}