import {
  Button,
  ButtonGroup,
  Center,
  Grid,
  GridItem,
  Modal,
  ModalOverlay,
  ModalContent,
  ModalHeader,
  ModalCloseButton,
  ModalFooter,
  FormControl,
  FormLabel,
  Input,
} from "@chakra-ui/react";

import { ArrowRightIcon, ArrowLeftIcon } from "@chakra-ui/icons";
import { FaUserEdit, FaRegEye } from "react-icons/fa";
import { ReviewEditor } from "./ReviewEditor";
import { ReviewLabelsOverview } from "./ReviewLabelsOverview";
import { Feature } from "flagged";
import { ProgressBar } from "./ProgressBar";
import { downloadBlob, parseJSONReviews } from "../utils/files";
import {
  ANNOTATIONS,
  CUSTOM_USAGE_OPTIONS,
  METADATA,
  LABELLING_TOOL,
  IS_FLAGGED,
  IS_GOLDEN_DATASET,
  PREDICTED_USAGE_OPTIONS,
  CONTAINS_MORE_USAGE_OPTIONS,
  IS_LABELLED,
} from "../utils/labelKeys";

const React = require("react");
const { JSONUpload } = require("./JSONUpload");
const { Timer } = require("timer-node");

const timer = new Timer({ label: "review-inspection-timer" });

const relevantMetadataKeys = [
  ANNOTATIONS,
  CUSTOM_USAGE_OPTIONS,
  IS_FLAGGED,
  IS_GOLDEN_DATASET,
  PREDICTED_USAGE_OPTIONS,
  CONTAINS_MORE_USAGE_OPTIONS,
  IS_LABELLED,
];

function SetEditLabelIdModal({ onSave, value, isOpen, onClose }) {
  return (
    <>
      <Modal isOpen={isOpen} onClose={onClose}>
        <ModalOverlay />
        <ModalContent>
          <ModalHeader>Create New Label</ModalHeader>
          <ModalCloseButton />
          <FormControl>
            <FormLabel>Please enter your label ID</FormLabel>
            <Input
              value={value}
              onChange={(event) => onSave(event.target.value)}
            />
          </FormControl>

          <ModalFooter>
            <Button
              colorScheme="blue"
              mr={3}
              onClick={() => {
                onClose();
              }}
            >
              Create label
            </Button>
          </ModalFooter>
        </ModalContent>
      </Modal>
    </>
  );
}

export class Labeller extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      reviews: props.reviews || [],
      reviewIndex: 0,
      maxReviewIndex: 0,
      editLabelId: null,
      isInViewMode: true,
      isSetEditLabelIdModalOpen: false,
    };
  }

  loadJSONReviews = async (e) => {
    const jsonData = await parseJSONReviews(e);
    this.setState({
      version: jsonData.version,
      reviews: jsonData.reviews,
      reviewIndex: 0,
      maxReviewIndex: jsonData.maxReviewIndex,
    });
    timer.start();
  };

  availableLabelIds = (reviews) => {
    const availableLabelIds = new Set();
    reviews = reviews ? reviews : this.state.reviews;
    reviews.forEach((review) => {
      Object.keys(review.labels).forEach((labelId) =>
        availableLabelIds.add(labelId)
      );
    });
    return Array.from(availableLabelIds);
  };

  defaultLabelId = (reviews) => {
    return this.availableLabelIds(reviews)[0];
  };
  saveLabel = (key, data) => {
    const reviews = [...this.state.reviews];
    if (
      reviews[this.state.reviewIndex].labels[this.state.editLabelId] ===
      undefined
    ) {
      reviews[this.state.reviewIndex].labels[this.state.editLabelId] = {
        usageOptions: [],
        annotations: [],
      };
    }
    const label =
      reviews[this.state.reviewIndex].labels[this.state.editLabelId];
    label[key] = data;
    label.usageOptions = label[CUSTOM_USAGE_OPTIONS] || [];

    if (label[ANNOTATIONS].length > 0) {
      label.usageOptions = label.usageOptions.concat(
        label[ANNOTATIONS].map((annotation) => annotation.tokens.join(" "))
      );
    }
    this.setState({ reviews: reviews });
  };

  exportReviewsToJSON = () => {
    const formatReviewsJSONV3 = (reviews) => {
      const formattedReviews = structuredClone(reviews).map((review) => {
        Object.values(review.labels).forEach((label) => {
          if (
            IS_LABELLED in label === false ||
            label[IS_LABELLED] ||
            label[ANNOTATIONS].size > 0
          ) {
            // update usageOptions list
            if (CUSTOM_USAGE_OPTIONS in label) {
              label[CUSTOM_USAGE_OPTIONS].forEach((customUsageOption) => {
                if (!label["usageOptions"].includes(customUsageOption)) {
                  label["usageOptions"].push(customUsageOption);
                }
              });
            }
            if (ANNOTATIONS in label) {
              label[ANNOTATIONS].forEach((annotation) => {
                var usageOption = annotation["tokens"].join(" ");
                if (label["usageOptions"].includes(usageOption) === false) {
                  label["usageOptions"].push(usageOption);
                }
              });
            }
            if ("metadata" in label === false) {
              // add fields required in v3 format for this newly created label
              label["createdAt"] = new Date().toISOString();
              label["scores"] = {};
              label["datasets"] = {};
              label[METADATA] = { labellingTool: {} };
              delete label[IS_LABELLED];
            }

            // update metadata values
            relevantMetadataKeys.forEach((metadataKey) => {
              if (label[metadataKey] !== undefined) {
                label[METADATA][LABELLING_TOOL][metadataKey] = structuredClone(
                  label[metadataKey]
                );
              }
              delete label[metadataKey];
            });
            delete review.inspectionTime;
          }
        });

        // remove what is not specified JSON v3 format
        delete review.label;

        return review;
      });
      const reviewsDict = Object.fromEntries(
        formattedReviews.map((review) => [review["review_id"], review])
      );
      // remove review ID from the attributes to comply with JSON format v3
      Object.values(reviewsDict).forEach((review) => {
        delete review["review_id"];
        return review;
      });

      return reviewsDict;
    };
    const toJsonBlob = (data) => {
      return new Blob(
        [
          // ensure special characters are translated back into how they were in the original (e.g. ® -> \u00ae8 )
          JSON.stringify(data, (key, value) =>
            typeof value === "string"
              ? value.replace(
                  /[\u007f-\uffff]/g,
                  (c) =>
                    "\\u" + ("0000" + c.charCodeAt(0).toString(16)).slice(-4)
                )
              : value
          ).replace(/\\\\u/g, "\\u"),
        ],
        {
          type: "text/plain;charset=utf-8",
        }
      );
    };
    const reviews = formatReviewsJSONV3(this.state.reviews);

    downloadBlob(
      toJsonBlob({
        version: this.state.version,
        reviews: reviews,
      }),
      "my_data.json"
    );
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
    const res = await fetch("/api/saveLabels", {
      headers: {
        "Content-Type": "application/json",
      },
      method: "POST",
      body: JSON.stringify({
        run: this.props.run,
        sampleFileName: this.props.sampleFileName,
        labels: reviewState,
      }),
    });

    if (res.status === 200) {
      window.location.replace(
        `/label/thank-you?run=${encodeURIComponent(this.props.run)}`
      );
    }
    console.log(res);
  };

  convertToInternalLabellingFormat = (review) => {
    var reviewLabel = review.labels[this.state.editLabelId] || {};

    const singleLabelReview = structuredClone(review);
    singleLabelReview.label = review.labels[this.state.editLabelId] || {
      [ANNOTATIONS]: [],
      [CUSTOM_USAGE_OPTIONS]: [],
    };
    delete singleLabelReview.labels;

    if (this.state.editLabelId in review.labels === false) {
      return singleLabelReview;
    }

    if (METADATA in reviewLabel && LABELLING_TOOL in reviewLabel[METADATA]) {
      // if we have labellingTool metadata, the review is still in JSON v3 format
      var labellingToolMetadata = reviewLabel[METADATA][LABELLING_TOOL];

      if (CUSTOM_USAGE_OPTIONS in singleLabelReview.label === false) {
        singleLabelReview.label[CUSTOM_USAGE_OPTIONS] = [
          ...labellingToolMetadata[CUSTOM_USAGE_OPTIONS],
        ];
      }
      if (ANNOTATIONS in singleLabelReview.label === false) {
        singleLabelReview.label[ANNOTATIONS] = [
          ...labellingToolMetadata[ANNOTATIONS],
        ];
      }
    } else {
      // use "usageOptions" as custom usage options - no annotations available
      singleLabelReview.label[ANNOTATIONS] =
        singleLabelReview.label[ANNOTATIONS] || [];
      singleLabelReview.label[CUSTOM_USAGE_OPTIONS] =
        singleLabelReview.label[CUSTOM_USAGE_OPTIONS] || [];
    }

    return singleLabelReview;
  };

  filteredReviewsByLabelId = (labelId) => {
    return this.state.reviews.map((review) => {
      if (labelId in review.labels) {
        return review.labels[labelId];
      } else {
        return null;
      }
    });
  };

  onChangeeditLabelId = (e) => {
    this.setState({ editLabelId: e.target.value });
  };

  navigateToNext = () => {
    this.updateInspectionTime();

    this.setState({
      reviewIndex: this.state.reviewIndex + 1,
      maxReviewIndex: Math.max(
        this.state.reviewIndex + 1,
        this.state.maxReviewIndex
      ),
    });
  };

  navigateToPrevious = () => {
    if (this.state.reviewIndex > 0) {
      this.updateInspectionTime();
      this.setState({ reviewIndex: this.state.reviewIndex - 1 });
    }
  };
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
                    <ButtonGroup gap="2">
                      <SetEditLabelIdModal
                        onSave={(newValue) =>
                          this.setState({ editLabelId: newValue })
                        }
                        value={this.state.editLabelId}
                        isOpen={this.state.isSetEditLabelIdModalOpen}
                        onClose={() =>
                          this.setState({ isSetEditLabelIdModalOpen: false })
                        }
                      />

                      <Button
                        onClick={this.navigateToPrevious}
                        leftIcon={<ArrowLeftIcon />}
                        size="lg"
                        colorScheme="gray"
                        isDisabled={this.state.reviewIndex === 0}
                      >
                        Previous
                      </Button>
                      <Button
                        type="submit"
                        onClick={this.navigateToNext}
                        rightIcon={<ArrowRightIcon />}
                        size="lg"
                        isDisabled={
                          this.state.reviewIndex ===
                          this.state.reviews.length - 1
                        }
                      >
                        Next
                      </Button>
                      {this.state.isInViewMode ? (
                        <Button
                          size="lg"
                          onClick={() => {
                            if (this.state.editLabelId === null) {
                              this.setState({
                                isSetEditLabelIdModalOpen: true,
                              });
                            }
                            this.setState({
                              isInViewMode: false,
                            });
                          }}
                        >
                          <FaUserEdit />
                        </Button>
                      ) : (
                        <Button
                          size="lg"
                          onClick={() => {
                            this.setState({
                              isInViewMode: true,
                            });
                          }}
                        >
                          <FaRegEye />
                        </Button>
                      )}
                      {exportButton}
                    </ButtonGroup>
                  </Feature>
                  <Feature name="dynamicLabelling">
                    <ButtonGroup gap="2">
                      <Button
                        colorScheme="teal"
                        size="lg"
                        onClick={this.submitToS3}
                        isDisabled={this.props.sampleFileName === undefined}
                      >
                        Submit
                      </Button>
                    </ButtonGroup>
                  </Feature>
                </>
              }
            />
            {this.state.isInViewMode ? (
              <ReviewLabelsOverview
                review={this.state.reviews[this.state.reviewIndex]}
              />
            ) : (
              <ReviewEditor
                review={this.convertToInternalLabellingFormat(
                  this.state.reviews[this.state.reviewIndex]
                )}
                saveLabel={this.saveLabel}
                isDisabled={false}
                editLabelId={this.state.editLabelId}
              />
            )}
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
