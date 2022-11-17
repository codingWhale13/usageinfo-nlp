import React from 'react';
import { IconBase } from 'react-icons/lib';
const { Tag, TagLabel, TagRightIcon, Input, InputGroup, InputRightAddon, Button, IconButton, TagCloseButton, InputRightElement} = require('@chakra-ui/react');
const { CloseIcon, AddIcon, ArrowBackIcon } = require('@chakra-ui/icons');
export function RawUsageOptionTag({usageOption, deleteUsageOption, deleteReplacementClassesMapping, replacementClasses}){   
    const hasReplacementTag = replacementClasses.has(usageOption);
    
    return  (<Tag 
                colorScheme='green'
                variant='solid'
                size='md'
            >
        <TagLabel> {hasReplacementTag ? replacementClasses.get(usageOption) : usageOption}</TagLabel>
        {
            hasReplacementTag ?
            <TagRightIcon as={ArrowBackIcon} onClick={() => deleteReplacementClassesMapping(usageOption)}/> :
            <TagCloseButton size={"lg"} as={CloseIcon} onClick={() => deleteUsageOption(usageOption)} />
        }
        
    
    </Tag>);
}

export function UsageOptionTag({annotation, customUsageOption, replacementClasses, deleteAnnotation, deleteCustomUsageOption, deleteReplacementClassesMapping}){
    let deleteUsageOption, usageOption;

    if(customUsageOption){
        usageOption = customUsageOption;
        deleteUsageOption =  () => {deleteCustomUsageOption(customUsageOption)};
    }
    else if(annotation){
        usageOption = annotation.tokens.join(' ');
        deleteUsageOption = () => {deleteAnnotation(annotation)};
    }

  
    return <RawUsageOptionTag
            usageOption={usageOption}
            deleteUsageOption={deleteUsageOption}
            deleteReplacementClassesMapping={deleteReplacementClassesMapping}
            replacementClasses={replacementClasses}
        >
     </RawUsageOptionTag>;

}

export function CustomUsageOptionFormTag({onSave}){
    const saveNewCustomUsageOption = (e) => {
        e.preventDefault();
        const newCustomUsageOption = e.target.custom_usage_option.value;
        if(newCustomUsageOption){
            onSave(newCustomUsageOption);
        }
        //Reset form input
        e.target.custom_usage_option.value= '';
    };



    const formRef = React.createRef();

    return <Tag p={0}>
        <form onSubmit={saveNewCustomUsageOption} ref={formRef}>
            <InputGroup size='md'>
                <Input
                    name='custom_usage_option'
                    placeholder='Add new usage option'
                />
                <InputRightElement  p={0} children={
                    <IconButton type='submit' h='2rem' size='sm' icon={<AddIcon />} ></IconButton>
                }/>
            </InputGroup>
            
        </form>
        
    </Tag>;
}