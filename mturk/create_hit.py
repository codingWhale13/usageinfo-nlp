from liquid import Template
import xml.etree.ElementTree as ET
import xml.sax.saxutils as saxutils

def create_hit_question_html(html_path, source, metadata):
    with open(html_path) as file:
        html_page = file.read()
        template = Template(html_page)
        return template.render(source=source,metadata=metadata)


def create_hit_question_xml( source, metadata, html_path='build/index.html'):
    question = ET.Element("HTMLQuestion")
    question.set("xmlns","http://mechanicalturk.amazonaws.com/AWSMechanicalTurkDataSchemas/2011-11-11/HTMLQuestion.xsd")
    question_html = ET.SubElement(question, "HTMLContent")
    question_frame_height = ET.SubElement(question, "FrameHeight")
    #If you set the value to 0, your HIT will automatically resize to fit within the Worker's browser window.
    question_frame_height.text = '0'
    question_html.text = saxutils.escape(create_hit_question_html(html_path, source, metadata))

    return ET.tostring(question, encoding='unicode', method='xml')

if __name__ == '__main__':  
    xml = create_hit_question_xml('','')