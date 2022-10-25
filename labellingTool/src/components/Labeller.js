import { ReviewLabel } from './ReviewLabel';
import { Button } from '@chakra-ui/react'
const React = require('react');
const {CSVUpload} = require('./CSVUpload');
const Papa = require('papaparse');

export class Labeller extends React.Component{
    constructor(props){
        super(props);

        this.state = {
            reviews: [],
            labels: [],
            reviewIndex : 0
        };
    }

    parseReviews = (e) => {
        Papa.parse(e.target.files[0], {
            header: true,
            skipEmptyLines: true,
            delimiter: '\t',
            complete: (csv) => {
                const reviews = csv.data;
                this.setState({reviews: reviews});
            
            },
            error : (error) => {
                console.error(error)
            }
        });
    }

    saveReviewLabels = (labels, i) => {
        const reviews = [...this.state.reviews];
        reviews[i].label = labels;
        this.setState({reviews: reviews});
    }

    exportLabelsToCSV = () => {
        const csv = Papa.unparse(this.state.reviews,{delimiter:'\t'});
        console.log('csv:',csv);
        var blob = new Blob([csv], { type: 'text/tsv;charset=utf-8;' });
        let encodedUri = URL.createObjectURL(blob);
        let link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", "my_data2.tsv");
        document.body.appendChild(link); // Required for FF

        link.click();
    }

    

    render(){

        return (<>

            {this.state.reviews.length === 0 &&
                <>
                    <h1>Bitte ein .tsv Datei mit Reviews hochladen</h1>
                    <CSVUpload 
                        onUpload={this.parseReviews}
                    />
                 </>
            }
           

            
            {this.state.reviewIndex < this.state.reviews.length &&
                <>  
                   
                   <Button colorScheme='teal' size='lg' onClick={this.exportLabelsToCSV}>
            Export
        </Button>
                    <h5>{this.state.reviewIndex}</h5>
                    <ReviewLabel
                        review={this.state.reviews[this.state.reviewIndex]}
                        onSave={(labels) => this.saveReviewLabels(labels, this.state.reviewIndex)}
                        onFinished={() => this.setState({reviewIndex: this.state.reviewIndex + 1})}
                    />
                </>
            }
            {(this.state.reviewIndex !== 0 && this.state.reviewIndex >= this.state.reviews.length) &&
              <Button colorScheme='teal' size='lg' onClick={this.exportLabelsToCSV}>
              Export
          </Button>
            }
            
      </>);
    }


}