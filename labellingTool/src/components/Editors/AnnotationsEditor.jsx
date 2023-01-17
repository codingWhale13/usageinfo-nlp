import { ANNOTATIONS } from '../../utils/labelKeys';
import { Wrap, Heading } from '@chakra-ui/react';
import { UsageOptionTag } from '../UsageOptionTag';
export function AnnotationsEditor({
    annotations,
    saveLabel,
    saveCustomUsageOption,
  }) {
    const uniqueAnnotations = annotations => {
      return annotations.filter(
        (annotation, index) =>
          annotations
            .map(annotation => annotation.tokens.join(' '))
            .indexOf(annotation.tokens.join(' ')) === index
      );
    };
  
    const deleteAnnotation = annotation => {
      saveLabel(
        ANNOTATIONS,
        annotations.filter(
          annotationA =>
            annotationA.tokens.join(' ') !== annotation.tokens.join(' ')
        )
      );
    };
  
    const updateAnnotation = annotation => {
      return newCustomUsageOption => {
        if (newCustomUsageOption === '') {
          deleteAnnotation(annotation);
        } else {
          saveCustomUsageOption(newCustomUsageOption);
          deleteAnnotation(annotation);
        }
      };
    };
  
    return (
      <Wrap spacing={2} pt="2">
        <Heading as="h5" size="sm" paddingY={2}>
          Annotated usage options
        </Heading>
        {uniqueAnnotations(annotations).map(annotation => (
          <UsageOptionTag
            usageOption={annotation.tokens.join(' ')}
            key={annotation.tokens.join(' ')}
            onDeleteUsageOption={() => deleteAnnotation(annotation)}
            onUpdateUsageOption={updateAnnotation(annotation)}
          ></UsageOptionTag>
        ))}
      </Wrap>
    );
  }