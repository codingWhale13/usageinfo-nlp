import { Button, Progress, Heading, Container,   Stat,
    StatLabel,
    StatNumber,
    VStack, Spacer, ButtonGroup, Box } from '@chakra-ui/react'
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
        const blob = new Blob([csv], { type: 'text/tsv;charset=utf-8;' });
        this.downloadBlob(blob, 'my_data.tsv');   
    }

    

    render(){
        const reviewLabel = (this.state.reviews.length && this.state.reviewIndex < this.state.reviews.length)  ? this.state.reviews[this.state.reviewIndex].label : {};
        console.log(this.state, reviewLabel);

    
        const exportButtons = <>
             <Button colorScheme='teal' size='lg' onClick={this.exportLabelsToCSV}>
                Export to TSV
            </Button>
            <Button colorScheme='teal' size='lg' onClick={this.exportReviewsToJSON}>
                Export to JSON
            </Button>
        </>;

        return (<Container maxWidth='1300px'>
            
            {this.state.reviews.length === 0 &&
                <>
                    <h1>Bitte ein .tsv Datei mit Reviews hochladen</h1>
                    <CSVUpload 
                        onUpload={this.parseReviews}
                    />
                    <h1>Oder eine bereits gelabelled JSON Datei hochladen</h1>
                    <JSONUpload 
                        onUpload={this.parseJSONReviews}
                    />
                </>
            }
           {this.state.reviewIndex < this.state.reviews.length &&
                <>
                   <Card spacing={2} mb={2}>
                        <VStack spacing='1px' align='left'>
                        
                            <Box>
                                <Heading as='h5' size='md'>Label reviews</Heading>
                                <Stat>
                                    <StatNumber>{this.state.reviewIndex + 1}/{this.state.reviews.length}</StatNumber>
                                </Stat>
                            </Box>
                            <Spacer />
                            <Feature name="localLabelling">
                                <ButtonGroup gap='2'>
                                {exportButtons}
                                </ButtonGroup>
                            </Feature>
                            
                        </VStack>

                        <Progress value={((this.state.reviewIndex + 1) / this.state.reviews.length) * 100} />
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

                        isFlagged={reviewLabel ? reviewLabel.isFlagged : false}
                        annotations={reviewLabel ? reviewLabel.annotations : []} />
                </>
            }
            {(this.state.reviewIndex !== 0 && this.state.reviewIndex >= this.state.reviews.length) &&

            <ButtonGroup>
                <Button onClick={() => { this.setState({ reviewIndex: this.state.reviewIndex - 1 });}}>
                    Previous
                </Button>
                
                <Feature name="localLabelling">
                    {exportButtons}
                </Feature>
            </ButtonGroup>
            }

      </Container>);
    }


}