import { Feature } from "flagged";
import { annotationsToUsageOptions } from "../../utils/conversion";
import {
  ANNOTATIONS,
  CUSTOM_USAGE_OPTIONS,
  IS_LABELLED,
} from "../../utils/labelKeys";
import { getFeatureFlags } from "../../featureFlags";
const React = require("react");
const { TokenAnnotator } = require("react-text-annotate");
const { Select, Divider, Box } = require("@chakra-ui/react");

const tokenizeString = require("../../utils/tokenize");
const { POSITIVE_TAG, NEGATIVE_TAG } = require("../../utils/tags");

const TAG_COLORS = {
  [POSITIVE_TAG]: "#8afd8a",
  [NEGATIVE_TAG]: "#fc8c90",
};

const features = getFeatureFlags();

export class ReviewTokenEditor extends React.Component {
  static defaultProps = {
    isDisabled: true,
    annotations: [],
  };

  constructor(props) {
    super(props);
    this.state = {
      tag: POSITIVE_TAG,
    };
  }

  saveAnnotations = (annotations) => {
    const usageOptions = annotationsToUsageOptions(annotations);
    this.props.saveLabel(
      CUSTOM_USAGE_OPTIONS,
      this.props[CUSTOM_USAGE_OPTIONS].filter(
        (usageOptionA) => !usageOptions.includes(usageOptionA)
      )
    );
    this.props.saveLabel(ANNOTATIONS, annotations);
    this.props.saveLabel(IS_LABELLED, true);
  };

  render() {
    return (
      <Box>
        <form
          onSubmit={(e) => {
            e.preventDefault();
          }}
        >
          <Feature name="negativeUseCases">
            <Select
              onChange={(e) => this.setState({ tag: e.target.value })}
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
              textAlign: "left",
              minHeight: "1000px",
            }}
            tokens={tokenizeString(this.props.review_body)}
            value={features.ratePredictedUseCases ? [] : this.props.annotations}
            onChange={(value) => {
              if (!features.ratePredictedUseCases && !this.props.isDisabled) {
                this.saveAnnotations(value);
              }
            }}
            getSpan={(span) => ({
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
