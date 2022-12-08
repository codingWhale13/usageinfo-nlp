import {
    Progress,
    Heading,
    Stat,
    StatNumber,
    Spacer,
    Box,
    Flex
  } from '@chakra-ui/react';
import { Card } from './Elements'


export function ProgressBar({numberOfReviews, currentReviewIndex, extra}){
    return (<Card spacing={2} mb={2}>
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
        value={
        ((currentReviewIndex + 1) / numberOfReviews) *
        100
        }
    />
    </Card>);
}