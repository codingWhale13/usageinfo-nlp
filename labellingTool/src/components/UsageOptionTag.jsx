import React from 'react';
const { Tag, TagLabel, TagRightIcon, Input, InputGroup, InputRightAddon, Button} = require('@chakra-ui/react');
const { CloseIcon, AddIcon } = require('@chakra-ui/icons');
export function UsageOptionTag({usageOption, onDelete}){    
    return  (<Tag 
                colorScheme='green'
                variant='solid'
                size='md'
            >
        <TagLabel> {usageOption}</TagLabel>
        <TagRightIcon 
            as={CloseIcon}
            onClick={onDelete}
        />
    
    </Tag>);
}

export function AnnotationUsageOptionTag({annotation, onDelete}){
    const usageOption = annotation.tokens.join(' ');
    return <UsageOptionTag
        usageOption={usageOption}
        onDelete={onDelete}
    />;
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