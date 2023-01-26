import { Button, ButtonGroup } from '@chakra-ui/react';
import React, { Component } from 'react';
import { ProgressBar } from './ProgressBar';
import { Review } from './Review';
import { InstructionsAlertDialog } from './InstructionsAlertDialog';
import { ANNOTATIONS, CONTAINS_MORE_USAGE_OPTIONS, CUSTOM_USAGE_OPTIONS, PREDICTED_USAGE_OPTIONS, PREDICTED_USAGE_OPTIONS_VOTE, PREDICTED_USAGE_OPTION_LABEL } from '../utils/labelKeys';
const { Timer } = require('timer-node');
const timer = new Timer({ label: 'review-inspection-timer' });

const REVIEWS =
  process.env.NODE_ENV === 'production'
    ? {
      metadata: JSON.parse(document.getElementById('metadata').innerText),
      review_bodies: JSON.parse(document.getElementById('source').innerText)
    }
    : {
      metadata: [{
        product_title: 'Super splitting axe',
        product_category: 'Outdoor',
        [CUSTOM_USAGE_OPTIONS]: ["dsas", "nsdjfdsjkd"],
        [ANNOTATIONS]: [
          {
              "start": 41,
              "end": 42,
              "tokens": [
                  "sanitize"
              ],
              "tag": "",
              "color": "#8afd8a"
          },
          {
              "start": 7,
              "end": 9,
              "tokens": [
                  "floor",
                  "cleaners"
              ],
              "tag": "",
              "color": "#8afd8a"
          }
      ]
      },
      {
        product_title: 'hello World 2',
        product_category: 'Test 2',
        [CUSTOM_USAGE_OPTIONS]: ["dsas"],
        [ANNOTATIONS]: [
          {
              "start": 41,
              "end": 42,
              "tokens": [
                  "sanitize"
              ],
              "tag": "",
              "color": "#8afd8a"
          },
          {
              "start": 7,
              "end": 9,
              "tokens": [
                  "floor",
                  "cleaners"
              ],
              "tag": "",
              "color": "#8afd8a"
          }
      ]
      }],
      review_bodies: ['I like this axe for chopping wood and digging a fire pit.', 'Hello world wow 1']
    };

class MTurkReview extends Component {
  constructor(props) {
    super(props);

    this.state = {
      reviews: [],
      label: {
        [ANNOTATIONS]: [],
        [CUSTOM_USAGE_OPTIONS]: [],
        [PREDICTED_USAGE_OPTIONS]: [],
        [CONTAINS_MORE_USAGE_OPTIONS]: []
      },
      inspectionTimes: [],
      reviewIndex: 0
    };

    for (let index = 0; index < REVIEWS.review_bodies.length; index++) {
      this.state.reviews.push({
        review_body: REVIEWS.review_bodies[index],
        ...REVIEWS.metadata[index]
      });
      this.state.label[ANNOTATIONS].push([]);
      this.state.label[CUSTOM_USAGE_OPTIONS].push([]);
      this.state.label[CONTAINS_MORE_USAGE_OPTIONS].push(false);

      const annotatations = REVIEWS.metadata[index][ANNOTATIONS].map((annotation) => annotation.tokens.join(' '));
      let allUsageOptions = REVIEWS.metadata[index][CUSTOM_USAGE_OPTIONS];
      allUsageOptions = allUsageOptions.concat(annotatations);

      this.state.label[PREDICTED_USAGE_OPTIONS].push(allUsageOptions.map((label) => { return { [PREDICTED_USAGE_OPTION_LABEL]: label, [PREDICTED_USAGE_OPTIONS_VOTE]: NaN}}))

      this.state.inspectionTimes.push(0);
    }

    timer.start();
  }

  saveLabel = (key, data) => {
    const label = { ...this.state.label };
    console.log(key, data, label);

    label[key][this.state.reviewIndex] = data;
    this.setState({ label: label });
  };

  updateInspectionTime = (callback) => {
    const time = timer.ms();
    const inspectionTimes = [...this.state.inspectionTimes];
    inspectionTimes[this.state.reviewIndex] = inspectionTimes[this.state.reviewIndex] + time;
    this.setState({ inspectionTimes: inspectionTimes }, () => {
      if (callback) {
        callback();
      }
    });
    timer.clear().start();
  };

  render() {
    const index = this.state.reviewIndex;
    const review = this.state.reviews[index];

    const annotations = this.state.label[ANNOTATIONS][index];
    const predictedUsageOptions = this.state.label[PREDICTED_USAGE_OPTIONS][index];
    const customUsageOptions = this.state.label[CUSTOM_USAGE_OPTIONS][index];
    const containsMoreUsageOptions = this.state.label[CONTAINS_MORE_USAGE_OPTIONS][index];
    const isLastReview = this.state.reviewIndex === this.state.reviews.length - 1;

    return (
      <>
        <ProgressBar
          currentReviewIndex={this.state.reviewIndex}
          numberOfReviews={this.state.reviews.length}
          extra={
            <ButtonGroup gap="2">
              <InstructionsAlertDialog />
              <Button colorScheme="teal" size="lg"
                isDisabled={!isLastReview}
                onClick={
                  () => {
                    this.updateInspectionTime(() => document.querySelector('crowd-form').submit());
                  }}>
                Submit task
              </Button>
            </ButtonGroup>
          }
        />

        <Review
          review={{
            ...review,
            label: {
              [ANNOTATIONS]: annotations,
              [CUSTOM_USAGE_OPTIONS]: customUsageOptions,
              [PREDICTED_USAGE_OPTIONS]: predictedUsageOptions,
              [CONTAINS_MORE_USAGE_OPTIONS]: containsMoreUsageOptions
            },
          }}

          saveLabel={this.saveLabel}

          navigateToNext={() => {
            this.updateInspectionTime();
            this.setState({
              reviewIndex: this.state.reviewIndex + 1,
            });
          }}

          isPreviousDisabled={this.state.reviewIndex === 0}

          isNextDisabled={isLastReview}

          navigateToPrevious={() => {
            this.updateInspectionTime();
            this.setState({ reviewIndex: this.state.reviewIndex - 1 });
          }}

        />

        <pre hidden>{JSON.stringify({
          annotations: this.state.label[ANNOTATIONS],
          customUsageOptions: this.state.label[CUSTOM_USAGE_OPTIONS],
          predictedUsageOptions: this.state.label[PREDICTED_USAGE_OPTIONS],
          [CONTAINS_MORE_USAGE_OPTIONS]: this.state.label[CONTAINS_MORE_USAGE_OPTIONS],
          inspectionTimes: this.state.inspectionTimes
        }, null, 2)}</pre>
      </>
    );
  }
}

export default MTurkReview;
