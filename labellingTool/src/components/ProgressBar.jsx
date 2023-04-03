import {
  Progress,
  Heading,
  Stat,
  StatNumber,
  Spacer,
  Box,
  Flex,
} from '@chakra-ui/react';
import { CustomCard } from './Elements';

export function ProgressBar({ numberOfReviews, currentReviewIndex, extra }) {
  return (
    <CustomCard spacing={2} mb={2}>
      <Flex>
        <Box>
          <Heading as="h5" size="md">
            Label reviews
          </Heading>
          <Stat>
            <StatNumber>
              {currentReviewIndex + 1}/{numberOfReviews}
            </StatNumber>
          </Stat>
        </Box>
        <Spacer />
        {extra}
      </Flex>

      <Progress
        mt={1}
        value={((currentReviewIndex + 1) / numberOfReviews) * 100}
      />
    </CustomCard>
  );
}
