import { Button, ButtonGroup, Center, Grid, GridItem, Select } from '@chakra-ui/react';
import { Review } from './Review';
import { Feature } from 'flagged';
import { ProgressBar } from './ProgressBar';
import { downloadBlob, parseJSONReviews } from '../utils/files';
import { ANNOTATIONS, CUSTOM_USAGE_OPTIONS } from '../utils/labelKeys';
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
      selectedLabelId: null,
    };
  }

  loadJSONReviews = async (e) => {
    const jsonData = await parseJSONReviews(e);
    this.setState({
      reviews: jsonData.reviews,
      reviewIndex: 0,
      maxReviewIndex: jsonData.maxReviewIndex,
      selectedLabelId: this.defaultLabelId(jsonData.reviews)
    });
    timer.start();
  };

  availableLabelIds = (reviews) => {
    const availableLabelIds = new Set();
    reviews = reviews ? reviews : this.state.reviews;
    reviews.forEach(review => {
      Object.keys(review.labels).forEach(labelId => availableLabelIds.add(labelId))
    });
    return Array.from(availableLabelIds);
  }

  defaultLabelId = (reviews) => {
    return this.availableLabelIds(reviews)[0];
  }
  saveLabel = (key, data) => {
    const reviews = [...this.state.reviews];
    reviews[this.state.reviewIndex].labels[this.state.selectedLabelId][key] = data;
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

    if (res.status === 200) {
      window.location.replace(`/label/thank-you?run=${encodeURIComponent(this.props.run)}`);
    }
    console.log(res);
  }

  convertToInternalLabellingFormat = (review) => {
    if (this.state.selectedLabelId in review.labels === false) {
      return null;
    }
    const singleLabelReview = structuredClone(review);
    console.log(this.state.selectedLabelId);
    singleLabelReview.label = review.labels[this.state.selectedLabelId] || {};

    if (CUSTOM_USAGE_OPTIONS in singleLabelReview.label === false) {
      singleLabelReview.label[CUSTOM_USAGE_OPTIONS] = [...review.labels[this.state.selectedLabelId].usageOptions];
    }
    if (ANNOTATIONS in singleLabelReview.label === false) {
      singleLabelReview.label[ANNOTATIONS] = [];
    }

    delete singleLabelReview.labels;
    return singleLabelReview;
  }

  filteredReviewsByLabelId = (labelId) => {
    return this.state.reviews.map((review) => {
      if (labelId in review.labels) {
        return review.labels[labelId];
      }
      else {
        return null
      }
    });
  }

  onChangeSelectedLabelId = (e) => {
    this.setState({ selectedLabelId: e.target.value });
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
                  <Select value={this.state.selectedLabelId} onChange={this.onChangeSelectedLabelId}>
                    {this.availableLabelIds().map((labelId) => (
                      <option value={labelId} key={labelId}>
                        {labelId}
                      </option>
                    ))}

                  </Select>
                  <Feature name="localLabelling">
                    <ButtonGroup gap="2">{exportButton}</ButtonGroup>
                  </Feature>
                  <Feature name="dynamicLabelling">
                    <ButtonGroup gap="2">
                      <Button colorScheme="teal" size="lg" onClick={this.submitToS3} isDisabled={this.props.sampleFileName === undefined}>
                        Submit
                      </Button>
                    </ButtonGroup>
                  </Feature>
                </>
              }
            />

            <Review
              review={this.convertToInternalLabellingFormat(this.state.reviews[this.state.reviewIndex])}
              saveLabel={this.saveLabel}

              isPreviousDisabled={this.state.reviewIndex === 0}

              isNextDisabled={this.state.reviewIndex === this.state.reviews.length - 1}

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
