import React, { Component } from 'react';
import { Review } from './Review';

const REVIEW =
  process.env.NODE_ENV === 'production'
    ? {
      metadata: JSON.parse(document.getElementById('metadata').innerHTML),
      review_body: document.getElementById('source').innerHTML
    }
    : {
      metadata: {
        product_title: 'hello World',
        product_category: 'Test'
      },
      review_body: 'Hello world wow 2',
      };
class MTurkReview extends Component {
  constructor(props) {
    super(props);

    this.state = {
      review: {review_body: REVIEW.review_body, ...REVIEW.metadata},
      annotations: [],
      customUsageOptions: [],
    };
  }

  render() {
    return (
      <>
        <Review
          review={{
            ...this.state.review,
            label: {
              annotations: this.state.annotations,
              customUsageOptions: this.state.customUsageOptions,
              replacementClasses: new Map(),
            },
          }}
          onSaveAnnotations={annotations => {
            console.log(annotations);
            this.setState({ annotations: annotations });
          }}
          onSaveCustomUsageOptions={customUsageOptions => {
            console.log('New custom usage option', customUsageOptions);
            this.setState({ customUsageOptions: customUsageOptions });
          }}
        />

        <pre hidden>{JSON.stringify(this.state, null, 2)}</pre>
      </>
    );
  }
}

export default MTurkReview;
