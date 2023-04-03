const { Box } = require("@chakra-ui/react");

export function CustomCard(props) {
  return (
    <Box borderWidth="1px" borderRadius="lg" boxShadow="md" p={3} mb={6}>
      {props.children}
    </Box>
  );
}
