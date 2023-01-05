import {
    Button,
    ButtonGroup,
    Center,
    Grid,
    GridItem,
  } from '@chakra-ui/react';
  import { Review } from './Review';
  import { Feature } from 'flagged';
import { ProgressBar } from './ProgressBar';
  
  const React = require('react');
  const { CSVUpload } = require('./CSVUpload');
  const { JSONUpload } = require('./JSONUpload');
  const { Timer } = require('timer-node');
  
  const Papa = require('papaparse');
  
  const timer = new Timer({label: 'review-inspection-timer'});
  export class Labeller extends React.Component {
    constructor(props) {
      super(props);
  
      this.state = {
        reviews: [],
        reviewIndex: 0,
        maxReviewIndex: 0,
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
                        };
                    review.inspectionTime = 0;
                }
                this.setState( {reviews: reviews });
            },
            error: (error) => {
                console.error(error)
            }
        });
        timer.start();
    }

    parseJSONReviews = async (e) => {
        const file = e.target.files[0];
        const jsonData = JSON.parse(await file.text());

        this.setState({
            reviews: jsonData.reviews,
            reviewIndex: jsonData.maxReviewIndex,
            maxReviewIndex: jsonData.maxReviewIndex
        });
        timer.start();
    }

    saveReviewFlag = (isFlagged, i) => {
      const reviews = [...this.state.reviews];
      reviews[i].label.isFlagged = isFlagged;
      this.setState({ reviews: reviews });
    };

    saveAnnotations = (annotations, i) => {
      const reviews = [...this.state.reviews];
      reviews[i].label.annotations = annotations;
      this.setState({ reviews: reviews });
    };

    saveCustomUsageOptions = (customUsageOptions, i) => {
        const reviews = [...this.state.reviews];
        reviews[i].label.customUsageOptions = customUsageOptions;
        this.setState({ reviews: reviews });
      };
  
    exportReviewsToJSON = () => {
      const reviewState = {
        reviews: this.state.reviews,
        maxReviewIndex: this.state.maxReviewIndex,
      };
      const jsonBlob = new Blob([JSON.stringify(reviewState)], {
        type: 'text/plain;charset=utf-8',
      });
      this.downloadBlob(jsonBlob, 'my_data.json');
    };
  
    downloadBlob = (blob, fileName) => {
      const encodedUri = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.setAttribute('href', encodedUri);
      link.setAttribute('download', fileName);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    };

    updateInspectionTime = () => {
      const time = timer.ms();
      const reviews = [...this.state.reviews];
      reviews[this.state.reviewIndex].inspectionTime = reviews[this.state.reviewIndex].inspectionTime + time;
      this.setState({ reviews: reviews });
      timer.clear().start();
  }
  
    render() {
      const reviewLabel =
        this.state.reviews.length &&
        this.state.reviewIndex < this.state.reviews.length
          ? this.state.reviews[this.state.reviewIndex]
          : {};
      console.log(this.state, reviewLabel);
  
      const exportButton = (
        <>
          <Button colorScheme="teal" size="lg" onClick={this.exportReviewsToJSON}>
            Export to JSON
          </Button>
        </>
      );
  
      return (
        <>
          {this.state.reviews.length === 0 && (
            <Center h="100vh">
              <Grid templateColumns="repeat(2, 1fr)">
                <GridItem>
                  <h1>Please upload a .tsv file:</h1>
                  <CSVUpload onUpload={this.parseReviews} />
                </GridItem>
                <GridItem>
                  <h1>Or upload an already labelled .json file:</h1>
                  <JSONUpload onUpload={this.parseJSONReviews} />
                </GridItem>
              </Grid>
            </Center>
          )}
          {this.state.reviewIndex < this.state.reviews.length && (
            <>
              <ProgressBar 
                currentReviewIndex={this.state.reviewIndex}
                numberOfReviews={this.state.reviews.length}
                extra={
                  <Feature name="localLabelling">
                    <ButtonGroup gap="2">{exportButton}</ButtonGroup>
                  </Feature>
                }
              />
  
              <Review
                review={this.state.reviews[this.state.reviewIndex]}

                onSaveAnnotations={ (annotations) => {
                  this.saveAnnotations(
                    annotations,
                    this.state.reviewIndex
                  );
                }}

                onSaveCustomUsageOptions={ (customUsageOptions) => {
                  this.saveCustomUsageOptions(
                    customUsageOptions,
                    this.state.reviewIndex
                  );
                }}

                onSaveFlag={ (isFlagged) => {
                  this.saveReviewFlag(isFlagged, this.state.reviewIndex);
                }}
                
                navigateToNext={() => {
                  this.updateInspectionTime();

                  this.setState({
                    reviewIndex: this.state.reviewIndex + 1,
                    maxReviewIndex: Math.max(
                      this.state.reviewIndex + 1,
                      this.state.maxReviewIndex
                    )
                  });
                }}

                navigateToPrevious={() => {
                  if (this.state.reviewIndex > 0) {
                        this.updateInspectionTime();
                        this.setState({ reviewIndex: this.state.reviewIndex - 1});
                  }
                }}
              />
            </>
          )}
          {this.state.reviewIndex !== 0 &&
            this.state.reviewIndex >= this.state.reviews.length && (
              <ButtonGroup>
                <Button
                  onClick={() => {
                    this.setState({ reviewIndex: this.state.reviewIndex - 1 });
                  }}
                >
                  Previous
                </Button>

                <Feature name="localLabelling">{exportButton}</Feature>
              </ButtonGroup>
            )}
        </>
      );
    }
  }
  