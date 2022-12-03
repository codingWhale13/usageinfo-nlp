import { Feature } from 'flagged';

const React = require('react');
const { TokenAnnotator } = require('react-text-annotate');
const { Select, Divider, Box } = require('@chakra-ui/react');

const tokenizeString = require('../utils/tokenize');
const { POSITIVE_TAG, NEGATIVE_TAG } = require('../utils/tags');

const TAG_COLORS = {
  [POSITIVE_TAG]: '#8afd8a',
  [NEGATIVE_TAG]: '#fc8c90',
};

export class ReviewTokenAnnotator extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      tag: POSITIVE_TAG,
    };
  }

  render() {
    return (
      <Box>
        <form
          onSubmit={e => {
            e.preventDefault();
          }}
        >
          <Feature name="negativeUseCases">
            <Select
              onChange={e => this.setState({ tag: e.target.value })}
              value={this.state.tag}
              spacing={20}
            >
              <option value={POSITIVE_TAG}>POSITIVE</option>
              <option value={NEGATIVE_TAG}>NEGATIVE</option>
            </Select>
            <Divider m={2} />
          </Feature>

          <TokenAnnotator
            style={{
              lineHeight: 1.5,
              textAlign: 'left',
              minHeight: '1000px',
            }}
            tokens={tokenizeString(this.props.review_body)}
            value={this.props.annotations}
            onChange={value => {
              this.props.onSaveAnnotations(value);
            }}
            getSpan={span => ({
              ...span,
              tag: this.state.tag,
              color: TAG_COLORS[this.state.tag],
            })}
          />
        </form>
      </Box>
    );
  }
}
