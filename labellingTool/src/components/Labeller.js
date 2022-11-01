import { ReviewLabel } from './ReviewLabel';
import { Button, Progress, Heading, Container,   Stat,
    StatLabel,
    StatNumber,
    StatHelpText,
    StatArrow,
    StatGroup, Flex, Spacer, ButtonGroup, Box } from '@chakra-ui/react'
import { Review } from './Review';
import { Card } from './Elements';
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

        return (<Container maxWidth='1300px'>

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
                   <Card spacing={2}>

                   <Flex minWidth='max-content' alignItems='center' gap='2'>
                    <Box>

                    
                   <Heading as='h2'>Label reviews</Heading>
                   <Stat>
                        <StatLabel>Labelled reviews</StatLabel>
                        <StatNumber>{this.state.reviewIndex}/{this.state.reviews.length}</StatNumber>
                        </Stat>
                     </Box>
                    <Spacer />
                    <ButtonGroup gap='2'>
                        <Button colorScheme='teal' size='lg' onClick={this.exportLabelsToCSV}>
                                    Export
                        </Button>
                    </ButtonGroup>
                    </Flex>
                        
                        
              
            <Progress value={(this.state.reviewIndex / this.state.reviews.length) * 100 } />
            </Card>
            
                    <Review  review={this.state.reviews[this.state.reviewIndex]}
                        onSave={(labels) => {
                            this.saveReviewLabels(labels, this.state.reviewIndex)
                            this.setState({reviewIndex: this.state.reviewIndex + 1})
                        }}
                    />
                </>
            }
            {(this.state.reviewIndex !== 0 && this.state.reviewIndex >= this.state.reviews.length) &&
              <Button colorScheme='teal' size='lg' onClick={this.exportLabelsToCSV}>
              Export
          </Button>
            }
            
      </Container>);
    }


}