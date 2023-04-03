import React from "react";
const {
  Tag,
  Input,
  InputGroup,
  IconButton,
  InputRightElement,
} = require("@chakra-ui/react");
const { CloseIcon, AddIcon } = require("@chakra-ui/icons");

export function UsageOptionTag({
  usageOption,
  onDeleteUsageOption,
  onUpdateUsageOption,
  isDisabled = true,
}) {
  const formRef = React.createRef();
  const [value, setValue] = React.useState(usageOption);
  const handleChange = (event) => setValue(event.target.value);

  const updateUsageOptionSubmission = (e) => {
    e.preventDefault();
    const newCustomUsageOption = e.target.custom_usage_option.value;
    if (newCustomUsageOption !== usageOption) {
      onUpdateUsageOption(newCustomUsageOption);
    }
  };

  return (
    <form
      onSubmit={updateUsageOptionSubmission}
      ref={formRef}
      isDisabled={isDisabled}
    >
      <Tag colorScheme="green" variant="solid" size="sm" p={0}>
        <InputGroup size="sm">
          <Input
            type="text"
            name="custom_usage_option"
            fontWeight={500}
            value={value}
            onChange={handleChange}
            onBlur={(e) => setValue(usageOption)}
            isDisabled={isDisabled}
          />
          <InputRightElement
            children={
              <IconButton
                m={0}
                colorScheme={"green"}
                color="black"
                h="1.5rem"
                size="xs"
                icon={<CloseIcon />}
                onClick={onDeleteUsageOption}
                isDisabled={isDisabled}
              ></IconButton>
            }
          />
        </InputGroup>
      </Tag>
    </form>
  );
}

export function CustomUsageOptionFormTag({ onSave, isDisabled = true }) {
  const saveNewCustomUsageOption = (e) => {
    e.preventDefault();
    const newCustomUsageOption = e.target.custom_usage_option.value;
    if (newCustomUsageOption) {
      onSave(newCustomUsageOption);
    }
    //Reset form input
    e.target.custom_usage_option.value = "";
  };

  const formRef = React.createRef();

  return (
    <Tag p={0}>
      <form onSubmit={saveNewCustomUsageOption} ref={formRef}>
        <InputGroup size="md">
          <Input
            name="custom_usage_option"
            placeholder="Add new usage option"
            isDisabled={isDisabled}
          />
          <InputRightElement
            p={0}
            children={
              <IconButton
                type="submit"
                h="2rem"
                size="sm"
                icon={<AddIcon />}
                isDisabled={isDisabled}
              ></IconButton>
            }
          />
        </InputGroup>
      </form>
    </Tag>
  );
}
