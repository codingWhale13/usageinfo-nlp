import { Feature } from 'flagged';

const React = require('react');
const { TokenAnnotator } = require('react-text-annotate');
const { Select,  Divider, Container, Box } =  require('@chakra-ui/react');

const tokenizeString = require('../utils/tokenize');
const {POSITIVE_TAG, NEGATIVE_TAG} = require('../utils/tags');

const TAG_COLORS = {
    [POSITIVE_TAG] : '#8afd8a',
    [NEGATIVE_TAG] : '#fc8c90'
};



export class ReviewTokenAnnotator extends React.Component{
 
   constructor(props){
        super(props);
        this.state = {
            tag: POSITIVE_TAG
        }
    }

    mergeAnnotations = (annotations) => {
        if(annotations.length === 0){
            return annotations;
        }
        annotations.sort((a, b) => a.start - b.start);
        const mergedAnnotations = [annotations[0]];
        let i = 0;
        let mergedIndex = 0;
        while (i < annotations.length - 1) {
            const currentSelection = mergedAnnotations[mergedIndex];
            const nextSelection = annotations[i + 1];
            if(currentSelection.end  === nextSelection.start && currentSelection.tag === nextSelection.tag){
                const mergedSelection = {
                    start: currentSelection.start,
                    end: nextSelection.end,
                    tokens: currentSelection.tokens.concat(nextSelection.tokens),
                    tag: currentSelection.tag,
                    color: currentSelection.color
                };
                mergedAnnotations[mergedIndex] = mergedSelection;
                i++;
                
            }
            else{
                mergedAnnotations.push(nextSelection);
                mergedIndex++;
                i++;
            }
        }
        return mergedAnnotations;
    }

    

    render(){
        return (<Box>
           
        <form
            onSubmit={(e) => {
                e.preventDefault();
            }}
        >   
        
            <Feature name="negativeUseCases">
                <Select
                    onChange={e => this.setState({ tag: e.target.value })}
                    value={this.state.tag}
                    spacing={20}
                >
                    <option value={POSITIVE_TAG}>POSITIVE</option>
                    <option value={NEGATIVE_TAG}>NEGATIVE</option>
                </Select>
                <Divider m={2}/>
            </Feature>
              
            <TokenAnnotator
                style={{
                    lineHeight: 1.5,
                    textAlign: 'left',
                    minHeight: '1000px'
                }}
                tokens={tokenizeString(this.props.review_body)}
                value={this.props.annotations}
                onChange={value => {
                    const mergedValue = this.mergeAnnotations(value);
                    this.props.onSaveAnnotations(mergedValue);
                }}
                getSpan={span => ({
                    ...span,
                    tag: this.state.tag,
                    color: TAG_COLORS[this.state.tag],
                })}
            />
           
            
      </form>
    </Box>);
    }
}