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
        onSave(newCustomUsageOption);
        //Reset form input
        e.target.custom_usage_option.value= '';
    };



    const formRef = React.createRef();

    return <Tag>
        <form onSubmit={saveNewCustomUsageOption} ref={formRef}>
            <InputGroup size='md'>
            <Input
                name='custom_usage_option'
                placeholder='Add new usage option'
            />
            <InputRightAddon children={
                   <Button type='submit' m={0} p={0} spacing={0}> < AddIcon /></Button>
            }/>
             
            </InputGroup>
            
        </form>
        
    </Tag>;
}