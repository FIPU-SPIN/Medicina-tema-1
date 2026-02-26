from graphviz import Digraph
import os
os.environ["PATH"] += r";C:\Program Files (x86)\Graphviz"

dot = Digraph(comment='LLM Interpretation Workflow', format='pdf')

dot.attr(rankdir='TB',  
         fontsize='10', 
         fontname='Helvetica',
         nodesep='0.5',
         ranksep='0.8')

node_attr = {
    'shape': 'box',
    'style': 'filled',
    'color': '#AED6F1',  
    'fontname': 'Helvetica',
    'fontsize': '11',
    'width': '2',
    'height': '0.8',
    'fixedsize': 'false',
    'dpi': '96',          
    'margin': '0.2,0.2'
}

dot.attr('node', **node_attr)
dot.attr(dpi='300')
dot.node('A', 'Clinical Reports (raw)')
dot.node('B', 'Extraction of Clinical Findings')
dot.node('C', 'Retrieval of Validated Medical References')
dot.node('D', 'Prompt Construction\n(finding + retrieved context)')
dot.node('E', 'LLM Interpretation Generation')
dot.node('F', 'Structured Output Storage\n(input, retrieval, interpretation, reference)')
dot.node('G', 'Automated & Manual Evaluation')
dot.attr('edge', color='#34495E', penwidth='2', )
dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG'])
dot.render('llm_workflow3', view=True)