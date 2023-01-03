import { Button, ButtonGroup } from '@chakra-ui/react';
import React, { Component } from 'react';
import { ProgressBar } from './ProgressBar';
import { Review } from './Review';
import { InstructionsAlertDialog } from './InstructionsAlertDialog';
const REVIEWS =
  process.env.NODE_ENV === 'production'
    ? {
      metadata: JSON.parse(document.getElementById('metadata').innerHTML),
      review_bodys: JSON.parse(document.getElementById('source').innerHTML)
    }
    : {
      metadata: [{
        product_title: 'Super splitting axe',
        product_category: 'Outdoor'
      },
      {
        product_title: 'hello World 2',
        product_category: 'Test 2'
      }],
      review_bodys: ['I like this axe for chopping wood and digging a fire pit.','Hello world wow 1']
      };
class MTurkReview extends Component {
  constructor(props) {
    super(props);

    this.state = {
      reviews: [],
      annotations: [],
      customUsageOptions: [],
      reviewIndex: 0
    };

    for (let index = 0; index < REVIEWS.review_bodys.length; index++) {
      this.state.reviews.push({
        review_body: REVIEWS.review_bodys[index],
        ...REVIEWS.metadata[index]
      });
      this.state.annotations.push([]);
      this.state.customUsageOptions.push([]);
    }
  }

  saveAnnotations = (reviewAnnotations, i) => {
    console.log(reviewAnnotations)
    const annotations = [...this.state.annotations];
    annotations[i] = reviewAnnotations;
    this.setState({ annotations: annotations });
  };

  saveCustomUsageOptions = (reviewCustomUsageOptions, i) => {
      const customUsageOptions = [...this.state.customUsageOptions];
      customUsageOptions[i] = reviewCustomUsageOptions;
      this.setState({ customUsageOptions: customUsageOptions });
  };

  render() {
    const index = this.state.reviewIndex;
    const review = this.state.reviews[index];
    const annotations = this.state.annotations[index];
    const customUsageOptions = this.state.customUsageOptions[index];

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
                disabled={!isLastReview}
                onClick={
              () => {
                document.querySelector('crowd-form').submit();
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
              annotations: annotations,
              customUsageOptions: customUsageOptions,
            },
          }}
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

          navigateToNext={() => {
            this.setState({
              reviewIndex: this.state.reviewIndex + 1,
            });
          }}

          isPreviousDisabled={this.state.reviewIndex === 0}
          isNextDisabled={isLastReview}
          navigateToPrevious={() => {
              this.setState({ reviewIndex: this.state.reviewIndex - 1});
          }}

        />

        <pre hidden>{JSON.stringify({
          annotations: this.state.annotations,
          customUsageOptions: this.state.customUsageOptions
        }, null, 2)}</pre>
      </>
    );
  }
}

export default MTurkReview;
