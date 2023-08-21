# Environment Variables
import os

import json
import string
# Kor Extraction Chains - DO NOT USE LLM EXTRACTION CHAIN
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text

# LLMs
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import initialize_agent, AgentType

from langchain.tools import Tool

from langchain.chat_models import ChatOpenAI
from langchain.chains import SimpleSequentialChain

#ArXiv
import arxiv
from scraper import get_arXiv_id

#WandB logger
import wandb

#make .env file with YOUR OpenAI API Key
#run source .env in terminal to load your own OpenAI API Key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
print(OPENAI_API_KEY) #check it loaded

def printOutput(output):
    # Prints chain output formatted nicely as dict
    print(json.dumps(output,sort_keys=True, indent=3))

def remove_punctuation(input_string):
    # Make a translation table that maps all punctuation characters to None
    translator = str.maketrans("", "", string.punctuation)

    # Apply the translation table to the input string
    result = input_string.translate(translator)

    return result

def get_abstract(input="input"):
    # takes input from your chain
    # identifies researcher and arXiv_id separately
    arXiv_id = input[0:10]
    researcher = input[11:len(input)]

    #formatting
    id_list = [arXiv_id]

    # search arXiv for mst recent paper
    search = arxiv.Search(
        id_list=id_list,
        max_results=1, #can change this to get more papers
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending
    )

    for result in search.results():

        # text splitter (for chunking, avoids token limit)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=15000, chunk_overlap=100)

        # split paper information into different documents
        docs = text_splitter.create_documents([result.summary])
        docs = docs[0]

        # TODO make this validation actually work
        # currently has issues thinking the argument is len 12, and author is len 10
        # even though they print the same thing
        """""
        arg = researcher[1:len(researcher)-1]

        authors = []
        for i in range(0, len(result.authors)):
            authors.append(remove_punctuation(str(result.authors[i])))
        if arg in authors:
            return f"This is a a paper by: {researcher}"
        else:
            continue
            #return f"This paper is not by our researcher."
        """""



    return docs

get_abstract_tool = Tool(
    name="get_abstract",
    func=get_abstract,
    description= "Useful to get the abstract of a research paper from arXiv. \
    there should always be two inputs, one which is the arXiv id and the other which is the name of researcher \
    "

)

# using Kor to find key concepts
skill_schema = Object(
    id="abstract",

    description="the abstract of a research paper",

    # concepts: change 'physics or mathematical' to whatever your research is in
    # fields: change 'physics' to your subject
    attributes=[
        Text(id="concepts",
             description="a list of key physics or mathematical concepts the research paper discusses"),
        Text(id="fields",
             description="a list of key fields of physics which encapsulate the concepts the research paper discusses")
    ],

    # you may want to change examples if you are looking at professors in a vastly different area
    examples=[
        ("Effective field theories (EFT) of dark energy (DE) — built to parameterise the properties of DE in an \
    agnostic manner — are severely constrained by measurements of the propagation speed of gravitational waves\
    (GW). However, GW frequencies probed by ground-based interferometers lie around the typical strong coupling\
    scale of the EFT, and it is likely that the effective description breaks down before even reaching that \
    scale. We discuss how this leaves the possibility that an appropriate ultraviolet completion of DE \
    scenarios, valid at scales beyond an EFT description, can avoid present constraints on the GW speed. \
    Instead, additional constraints in the lower frequency LISA band would be harder to escape, since the \
    energies involved are orders of magnitude lower. By implementing a method based on GW multiband detections,\
    we show indeed that a single joint observation of a GW150914-like event by LISA and a terrestrial \
    interferometer would allow one to constrain the speed of light and gravitons to match to within 10-15. \
    Multiband GW observations can therefore firmly constrain scenarios based on the EFT of DE, in a robust \
    and unambiguous way.",
        [{"concepts": ["Effective field theory", "Dark Energy", "Gravitational Waves"]},
         {"fields": ["Effective field theory", "Cosmology"]}]),

        ("String compactifications typically require fluxes, for example in order to stabilise moduli. Such fluxes, \
    when they thread internal dimensions, are topological in nature and take on quantised values. This poses \
    the puzzle as to how they could arise in the early universe, as they cannot be turned on incrementally. \
    Working with string inspired models in 6 and 8 dimensions, we show that there exist no-boundary solutions \
    in which internal fluxes are present from the creation of the universe onwards. The no-boundary proposal \
    can thus explain the origin of fluxes in a Kaluza-Klein context. In fact, it acts as a selection principle \
    since no-boundary solutions are only found to exist when the fluxes have the right magnitude to lead to an \
    effective potential that is positive and flat enough for accelerated expansion. Within the range of \
    selected fluxes, the no-boundary wave function assigns higher probability to smaller values of flux. Our \
    models illustrate how cosmology can act as a filter on a landscape of possible higher-dimensional solutions.",
         [{"concepts": ["String Compactification", "No-Boundary Solutions", "Cosmology"]},
          {"fields": ["String Theory", "Cosmology"]}])
    ]
)

#log into wandb yourself to see all the outputs in a table, and analyse if it goes wrong
#if you don't want to use it, put in offline mode or comment out these lines, and wandb.finish() at the end
wandb.init(project='PhD_AI', job_type="generation")
os.environ["LANGCHAIN_WANDB_TRACING"] = "true"

#make chain
#use gpt-4 if able
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", # cheaper but less reliable
    #model_name="gpt-4", # requires special key
    temperature=0,
    max_tokens=2000,
    openai_api_key=OPENAI_API_KEY
)

# find paper on arXiv for specific researcher
tools = [get_abstract_tool]
agent = initialize_agent(tools, llm,
                         agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                         verbose=True,
                         max_iterations=3,
                         handle_parsing_errors=True)

# find abstracts and return document part of the chain
paper_chain = agent

# read document and return key concepts part of the chain
skill_chain = create_extraction_chain(llm, skill_schema)

# combined chain which takes name and arXiv id and gives concepts and fields in their research
overall_chain = SimpleSequentialChain(
                  chains=[paper_chain, skill_chain],
                  verbose=True)

# change researcher name to whoever you want
researcher = "Andrew Tolley"
arXiv_id = str(get_arXiv_id(researcher)) #from scraper.py
print(arXiv_id)

output = overall_chain.run([researcher, arXiv_id])["data"]

printOutput(output)

wandb.finish() #comment out if not using wandb