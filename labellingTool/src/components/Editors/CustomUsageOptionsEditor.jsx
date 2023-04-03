import { CUSTOM_USAGE_OPTIONS } from '../../utils/labelKeys';
import { annotationsToUsageOptions } from '../../utils/conversion';
import { Heading, Wrap } from '@chakra-ui/react';
import { CustomUsageOptionFormTag, UsageOptionTag } from '../UsageOptionTag';

export function CustomUsageOptionsEditor({
  customUsageOptions,
  annotations,
  saveLabel,
  saveCustomUsageOption,
}) {
  const deleteCustomUsageOption = customUsageOption => {
    saveLabel(
      CUSTOM_USAGE_OPTIONS,
      customUsageOptions.filter(
        usageOptionA => usageOptionA !== customUsageOption
      )
    );
  };
  const updateCustomUsageOption = customUsageOption => {
    return updatedCustomUsageOption => {
      if (
        annotationsToUsageOptions(annotations).includes(
          updatedCustomUsageOption
        ) ||
        updatedCustomUsageOption === ''
      ) {
        deleteCustomUsageOption(customUsageOption);
      } else {
        saveLabel(
          CUSTOM_USAGE_OPTIONS,
          customUsageOptions
            .map(usageOptionA =>
              usageOptionA === customUsageOption
                ? updatedCustomUsageOption
                : usageOptionA
            )
            .filter(
              (usageOptionA, index, self) =>
                self.indexOf(usageOptionA) === index
            )
        ); // different from saveCustomUsageOption because we do not append to list
      }
    };
  };

  return (
    <>
      <Heading as="h5" size="sm" paddingY={2}>
        Custom usage options
      </Heading>

      <CustomUsageOptionFormTag onSave={saveCustomUsageOption} />
      <Wrap spacing={2} pt="5">
        {customUsageOptions.map(customUsageOption => (
          <UsageOptionTag
            usageOption={customUsageOption}
            key={customUsageOption}
            onDeleteUsageOption={() =>
              deleteCustomUsageOption(customUsageOption)
            }
            onUpdateUsageOption={updateCustomUsageOption(customUsageOption)}
          ></UsageOptionTag>
        ))}
      </Wrap>
    </>
  );
}
