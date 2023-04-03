import {
  Text,
  Tag,
  Divider,
  Stack,
  TagLabel,
  TagRightIcon,
} from "@chakra-ui/react";
import { StarIcon } from "@chakra-ui/icons";
import { CustomCard } from "./Elements";
import { FaThumbsUp, FaThumbsDown } from "react-icons/fa";
export function ReviewHeader({ review }) {
  const starTatingToColorScheme = {
    1: "red",
    2: "red",
    3: "orange",
    4: "green",
    5: "green",
  };
  return (
    <CustomCard>
      <Text
        fontWeight={"bold"}
        fontSize={"20px"}
        as="h3"
        size="md"
        textAlign="left"
        noOfLines={1}
      >
        {review.product_title}
      </Text>
      <Divider m={2} />
      <Stack direction={["column", "row"]}>
        <Tag size="lg" colorScheme="blue">
          {review.product_category}
        </Tag>
        <Tag size="lg" colorScheme="blue">
          {review.review_id}
        </Tag>
        <Tag
          size="lg"
          colorScheme={starTatingToColorScheme[review.star_rating]}
        >
          <TagLabel>{review.star_rating} / 5</TagLabel>
          <TagRightIcon>
            <StarIcon />
          </TagRightIcon>
        </Tag>

        <Tag size="lg" colorScheme="green">
          <TagLabel>
            {review.helpful_votes > 0 ? review.helpful_votes : 0}
          </TagLabel>
          <span style={{ "padding-left": "0.5rem" }}>
            <FaThumbsUp />
          </span>
        </Tag>

        <Tag size="lg" colorScheme="red">
          <TagLabel>{review.total_votes - review.helpful_votes}</TagLabel>
          <span style={{ "padding-left": "0.5rem" }}>
            <FaThumbsDown />
          </span>
        </Tag>
      </Stack>
    </CustomCard>
  );
}
