const React = require('react');
const Highlighter = require('web-highlighter');
const { Switch, Button, ButtonGroup } =  require('@chakra-ui/react');

export class ReviewLabel extends React.Component{
    constructor(props){
        super(props);

        this.setupHighlighter();
        this.reviewRef = React.createRef();

        this.state = {
            selections: [],
            selectionsWithSentiment: [],
            labelSentiment: false
        };
    }

    setupHighlighter = () => {
        this.highlighter = new Highlighter({
            exceptSelectors: ['h1','pre', 'code']
        });

        // auto-highlight selections
        this.highlighter.run();
        this.highlighter.removeAll();
    }

    resetSelection = () => {
        this.highlighter.removeAll();
        this.props.onSave([])
        this.setState({selections: [], selectionsWithSentiment: [], labelSentiment: false});
    }


    //web-highligther renders all highlighted text in spans. We write the review inside a paragraph, so we have to search for text inside spans
    indexOfSpanInParagraph = (p, text, searchStartPosition) => {
        let index = 0;
        for(let i = 0; i < p.childNodes.length; i++){
            let element = p.childNodes[i];
            if(element.nodeName === '#text'){
                
                index += element.length;
            }
            else if(element.nodeName === 'SPAN'){
                if(element.innerText === text && index >= searchStartPosition){
                    return index;
                }
                else{
                    
                    index += element.innerText.length;
                }
            }
        };

        return -1;
        
    }

    saveSelection = () => {
        const selections = [];
        let currentSearchPosition = 0;
        //get selections
        
        this.highlighter.getDoms().forEach(element => {
            if(element.textContent !== ''){
                const selectionIndex = this.indexOfSpanInParagraph(this.reviewRef.current, element.textContent, currentSearchPosition);
                const endIndex = selectionIndex + element.textContent.length - 1;
                selections.push({
                    text: element.textContent,
                    startIndex: selectionIndex,
                    endIndex: endIndex
                });
                currentSearchPosition = endIndex + 1;
            }
            
        }); 
        
        const mergedSelections = this.mergeSelections(selections);

        const seletionsWithSentiment = mergedSelections.map((selection) => { return {selection: selection, sentiment: 1}});
        return {
            selections: mergedSelections,
            selectionsWithSentiment: seletionsWithSentiment
        };

    }

    mergeSelections = (selections) => {
        if(selections.length === 0){
            return [];
        }
        const mergedSelections = [selections[0]];
        let i = 0;
        let mergedIndex = 0;
        while (i < selections.length - 1) {
            const currentSelection = mergedSelections[mergedIndex];
            const nextSelection = selections[i + 1];
            if(currentSelection.endIndex + 1 === nextSelection.startIndex){
                const mergedSelection = {
                    text: currentSelection.text + nextSelection.text,
                    startIndex: currentSelection.startIndex,
                    endIndex: nextSelection.endIndex
                };
                mergedSelections[mergedIndex] = mergedSelection;
                i++;
                
            }
            else{
                mergedSelections.push(nextSelection);
                mergedIndex++;
                i++;
            }
        }
        return mergedSelections;
    }
    
    componentDidUpdate(prevProps) {
        if(prevProps.review !== this.props.review){
            this.resetSelection();
            this.setupHighlighter()
        }
    }

    componentDidMount(){
        this.saveSelection();
    }

    selectionsToLabels = (selectionsWithSentiment) => {
        return selectionsWithSentiment.map((o) => [o.selection.startIndex, o.selection.endIndex, o.sentiment]);
    }
    saveLabels = () => {
        const {selectionsWithSentiment} = this.saveSelection();
        const labels = this.selectionsToLabels(selectionsWithSentiment);
        this.props.onSave(labels);
    }

    setSentiment = (sentiment, index) => {
        const selections = this.state.selectionsWithSentiment;
        if(selections[index].sentiment !== sentiment){
            selections[index].sentiment = sentiment;
            this.setState({selectionsWithSentiment: selections});
            this.props.onSave(this.selectionsToLabels(selections));
        }
    }

    startLabellingSentiment = () => {
        this.saveLabels();

        const {selections, selectionsWithSentiment} = this.saveSelection();
        this.setState({labelSentiment: true, selections: selections, selectionsWithSentiment: selectionsWithSentiment});
    }

    render(){
        return (<>
            {this.state.labelSentiment ?
                <>
                    {this.state.selections.map((selection, i) => 
                        <>  
                            <code>{selection.text}</code>
                        <Switch 
                                defaultChecked
                                onChange={(e) => this.setSentiment(e.target.checked ? 1 : 0, i)}
                        />
                        </>
                    )}
                    {this.state.selections.length === 0 &&
                        <code>No usage information selected</code>
                    }

                    <ButtonGroup gap='4'>
                    <Button onClick={this.resetSelection}>
                        Reset
                    </Button>
                        <Button
                            onClick={this.props.onFinished}
                            colorScheme='green'
                            >
                            Next
                        </Button>
                    </ButtonGroup>

                   
                </>
                :
                <>
                <p ref={this.reviewRef}>{this.props.review.review_body}</p>
                <ButtonGroup gap='4'>
                    <Button onClick={this.resetSelection}>
                    Reset
                </Button>
                    <Button colorScheme='green' onClick={this.startLabellingSentiment}>
                        Label sentiment
                    </Button>
                </ButtonGroup>
                </>
            }
           
            
            
        </>);
    }
}