import { Button, ButtonGroup, Center, Grid, GridItem } from '@chakra-ui/react';
import { Review } from './Review';
import { Feature } from 'flagged';
import { ProgressBar } from './ProgressBar';
import { downloadBlob, parseJSONReviews } from '../utils/files';
const React = require('react');
const { JSONUpload } = require('./JSONUpload');
const { Timer } = require('timer-node');

const timer = new Timer({ label: 'review-inspection-timer' });
export class Labeller extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      reviews: props.reviews || [],
      reviewIndex: 0,
      maxReviewIndex: 0,
    };
  }

  loadJSONReviews = async (e) => {
    const jsonData = await parseJSONReviews(e);
    this.setState({
      reviews: jsonData.reviews,
      reviewIndex: 0,
      maxReviewIndex: jsonData.maxReviewIndex,
    });
    timer.start();
  };

  saveLabel = (key, data) => {
    const reviews = [...this.state.reviews];
    reviews[this.state.reviewIndex].label[key] = data;
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
    downloadBlob(jsonBlob, 'my_data.json');
  };

  updateInspectionTime = () => {
    const time = timer.ms();
    const reviews = [...this.state.reviews];
    reviews[this.state.reviewIndex].inspectionTime =
      reviews[this.state.reviewIndex].inspectionTime + time;
    this.setState({ reviews: reviews });
    timer.clear().start();
  };

  submitToS3 = async () => {
    const reviewState = {
      reviews: this.state.reviews,
      maxReviewIndex: this.state.maxReviewIndex,
    };
    const res = await fetch('/api/saveLabels', {
      headers: {
        'Content-Type': 'application/json'
      },
      method: 'POST',
      body: JSON.stringify({
        run: this.props.run,
        sampleFileName: this.props.sampleFileName,
        labels: reviewState
      })
    });

    if(res.status === 200){
      window.location.replace(`/label/thank-you?run=${encodeURIComponent(this.props.run)}`);
    }
    console.log(res);
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
            <Grid templateColumns="repeat(1, 1fr)">
              <GridItem>
                <h1>Upload .json file:</h1>
                <JSONUpload onUpload={this.loadJSONReviews} />
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
                <>
                <Feature name="localLabelling">
                  <ButtonGroup gap="2">{exportButton}</ButtonGroup>
                </Feature>
                <Feature name="dynamicLabelling">
                  <ButtonGroup gap="2">
                    <Button colorScheme="teal" size="lg" onClick={this.submitToS3}>
                      Submit
                    </Button>
                  </ButtonGroup>
                </Feature>
                </>
              }
            />

            <Review
              review={this.state.reviews[this.state.reviewIndex]}
              saveLabel={this.saveLabel}
              
              isPreviousDisabled={this.state.reviewIndex === 0}
              
              navigateToNext={() => {
                this.updateInspectionTime();

                this.setState({
                  reviewIndex: this.state.reviewIndex + 1,
                  maxReviewIndex: Math.max(
                    this.state.reviewIndex + 1,
                    this.state.maxReviewIndex
                  ),
                });
              }}
              navigateToPrevious={() => {
                if (this.state.reviewIndex > 0) {
                  this.updateInspectionTime();
                  this.setState({ reviewIndex: this.state.reviewIndex - 1 });
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
