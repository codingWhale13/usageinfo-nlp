const React = require('react');
const { TokenAnnotator } = require('react-text-annotate');
const { Select, ButtonGroup, Button, Divider, Container } =  require('@chakra-ui/react');
const {StarIcon } = require('@chakra-ui/icons');

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

    resetAnnotation = () => {
        this.props.onSaveAnnotations([]);
    }

    render(){
        return (<Container>
           
        <form
            onSubmit={(e) => {
                e.preventDefault();
                this.props.navigateToNext();
            }}
        >   
        <ButtonGroup>
                <Button onClick={() => {
                   this.props.navigateToPrevious();
                }
                }>
                    Previous
                </Button>
                {this.props.isFlagged ? 
                    <Button colorScheme='red' onClick={() => this.props.onSaveFlag(false)}>
                    <StarIcon />
                    Remove flag
                </Button>
               
                : 
                <Button colorScheme='red' onClick={() => {
                    this.props.onSaveFlag(true);
                }}>
                    Flag for follow up
                </Button>
                }
                <Button onClick={this.resetAnnotation}>
                    Reset
                </Button>
                
                <Button type='submit' colorScheme='green'>
                    Next
                </Button>
            </ButtonGroup>

            <Divider m={2}/>
            <Select
                onChange={e => this.setState({ tag: e.target.value })}
                value={this.state.tag}
                spacing={20}
            >
                <option value={POSITIVE_TAG}>POSITIVE</option>
                <option value={NEGATIVE_TAG}>NEGATIVE</option>
            </Select>  
            <Divider m={2}/>
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
        </Container>);
    }
}