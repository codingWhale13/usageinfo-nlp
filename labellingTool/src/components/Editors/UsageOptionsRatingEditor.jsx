import {
  Heading,
  Wrap,
  Card,
  Text,
  CardBody,
  CardFooter,
  Button,
  ButtonGroup,
} from '@chakra-ui/react';
import { FaThumbsUp, FaThumbsDown } from 'react-icons/fa';
import {
  PREDICTED_USAGE_OPTIONS,
  PREDICTED_USAGE_OPTIONS_VOTE,
} from '../../utils/labelKeys';

function updateArrayAtIndex(list, index, newValue) {
  const newList = [...list];
  newList[index] = newValue;
  return newList;
}
function updateArrayAtIndexAtKey(array, index, key, newValue) {
  return updateArrayAtIndex(array, index, { ...array[index], [key]: newValue });
}

const GOOD_VOTE = 'good';
const BAD_VOTE = 'bad';

export function UsageOptionsRatingEditor({ predictedUsageOptions, saveLabel }) {
  return (
    <Wrap spacing={2} pt="2">
      <Heading as="h5" size="sm" paddingY={2}>
        Rate usage options
      </Heading>
      {predictedUsageOptions.map(({ label, vote }, i) => (
        <Card maxW="md" variant="outline" sx={{ '--card-padding': '0.5rem' }}>
          <CardBody>
            <Text>{label}</Text>
          </CardBody>

          <CardFooter
            justify="space-between"
            flexWrap="wrap"
            sx={{
              '& > button': {
                minW: '136px',
              },
            }}
          >
            <ButtonGroup>
              <ToggleButton
                text={'Upvote'}
                isOn={vote === GOOD_VOTE}
                onColor={'green'}
                leftIcon={<FaThumbsUp />}
                onClick={e =>
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
                text={'Downvote'}
                isOn={vote === BAD_VOTE}
                onColor={'red'}
                leftIcon={<FaThumbsDown />}
                onClick={e =>
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

function ToggleButton({ isOn, text, onColor, offColor, ...otherProps }) {
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
