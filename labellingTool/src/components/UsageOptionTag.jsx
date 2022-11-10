import React from 'react';
const { Tag, TagLabel, TagRightIcon, Input, InputGroup, InputRightAddon, Button} = require('@chakra-ui/react');
const { CloseIcon, AddIcon } = require('@chakra-ui/icons');
export function RawUsageOptionTag({usageOption, deleteUsageOption, deleteReplacementClassesMapping, replacementClasses}){    
    return  (<Tag 
                colorScheme='green'
                variant='solid'
                size='md'
            >
        <TagLabel> {replacementClasses.has(usageOption) ? replacementClasses.get(usageOption) : usageOption}</TagLabel>
        <TagRightIcon 
            as={CloseIcon}
            onClick={() => {
                deleteUsageOption();
                deleteReplacementClassesMapping(usageOption);
            }}
        />
    
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
            <InputRightAddon  p={0} children={
                   <Button type='submit' > < AddIcon /></Button>
            }/>
             
            </InputGroup>
            
        </form>
        
    </Tag>;
}