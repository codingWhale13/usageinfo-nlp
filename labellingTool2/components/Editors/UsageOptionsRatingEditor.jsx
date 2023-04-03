import {
  Heading,
  Wrap,
  Card,
  Text,
  CardBody,
  CardFooter,
  Button,
  ButtonGroup,
} from "@chakra-ui/react";
import { FaThumbsUp, FaThumbsDown, FaQuestionCircle } from "react-icons/fa";
import {
  PREDICTED_USAGE_OPTIONS,
  PREDICTED_USAGE_OPTIONS_VOTE,
} from "../../utils/labelKeys";

import {
  GOOD_VOTE,
  BAD_VOTE,
  QUESTION_VOTE,
} from "../../utils/voteDefinitions";
function updateArrayAtIndex(list, index, newValue) {
  const newList = [...list];
  newList[index] = newValue;
  return newList;
}

function updateArrayAtIndexAtKey(array, index, key, newValue) {
  return updateArrayAtIndex(array, index, { ...array[index], [key]: newValue });
}

export function UsageOptionsRatingEditor({ predictedUsageOptions, saveLabel }) {
  return (
    <Wrap spacing={2} pt="2">
      <Heading as="h5" size="sm" paddingY={2}>
        Rate usage options
      </Heading>

      {predictedUsageOptions.map(({ label, vote }, i) => (
        <Card
          maxW="100%"
          variant="outline"
          sx={{ "--card-padding": "0.5rem" }}
          key={({ label, vote }, i)}
        >
          <CardBody>
            <Text>{label}</Text>
          </CardBody>

          <CardFooter
            justify="space-between"
            flexWrap="wrap"
            sx={{
              "& > button": {
                minW: "136px",
              },
            }}
          >
            <ButtonGroup direction="row" spacing={3} align="center" size={"md"}>
              <ToggleButton
                text={"Upvote"}
                isOn={vote === GOOD_VOTE}
                onColor={"green"}
                leftIcon={<FaThumbsUp />}
                onClick={(e) =>
                  saveLabel(
                    PREDICTED_USAGE_OPTIONS,
                    updateArrayAtIndexAtKey(
                      predictedUsageOptions,
                      i,
                      PREDICTED_USAGE_OPTIONS_VOTE,
                      GOOD_VOTE
                    )
                  )
                }
              />

              <ToggleButton
                isOn={vote === QUESTION_VOTE}
                onColor={"yellow"}
                text={<FaQuestionCircle />}
                onClick={(e) =>
                  saveLabel(
                    PREDICTED_USAGE_OPTIONS,
                    updateArrayAtIndexAtKey(
                      predictedUsageOptions,
                      i,
                      PREDICTED_USAGE_OPTIONS_VOTE,
                      QUESTION_VOTE
                    )
                  )
                }
              />

              <ToggleButton
                text={"Downvote"}
                isOn={vote === BAD_VOTE}
                onColor={"red"}
                leftIcon={<FaThumbsDown />}
                onClick={(e) =>
                  saveLabel(
                    PREDICTED_USAGE_OPTIONS,
                    updateArrayAtIndexAtKey(
                      predictedUsageOptions,
                      i,
                      PREDICTED_USAGE_OPTIONS_VOTE,
                      BAD_VOTE
                    )
                  )
                }
              />
            </ButtonGroup>
          </CardFooter>
        </Card>
      ))}
    </Wrap>
  );
}

export function ToggleButton({ isOn, text, onColor, offColor, ...otherProps }) {
  if (isOn) {
    return (
      <Button colorScheme={onColor} {...otherProps}>
        {text}
      </Button>
    );
  } else {
    return (
      <Button colorScheme={offColor} {...otherProps}>
        {text}
      </Button>
    );
  }
}
