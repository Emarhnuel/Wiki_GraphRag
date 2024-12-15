from langchain.prompts import PromptTemplate
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.document_loaders import WikipediaLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Neo4jVector
from neo4j import GraphDatabase

def create_knowledge_graph(topic, llm, embeddings, graph):
    # Load more Wikipedia content
    loader = WikipediaLoader(query=topic, load_max_docs=5)  # Increased from 2 to 5
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Reduced chunk size for more granular splitting
    docs = text_splitter.split_documents(documents)

    # Clear the graph database
    cypher = """
      MATCH (n)
      DETACH DELETE n;
    """
    graph.query(cypher)

    # Define allowed nodes and relationships
    allowed_nodes = ["Person", "Organization", "Location", "Event", "Country",]
    allowed_relationships = ["RELATED_TO", "PART_OF", "LOCATED_IN", "PARTICIPATED_IN", ]

    # Transform documents into graph documents
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships
    ) 

    graph_documents = transformer.convert_to_graph_documents(docs)
    graph.add_graph_documents(graph_documents, include_source=True)

    return docs

def create_vector_index(driver, index_name, node_label, property_name, dimension):
    with driver.session() as session:
        # Drop existing index if it exists
        session.run(f"DROP INDEX {index_name} IF EXISTS")
        
        # Create new index
        session.run(f"""
        CALL db.index.vector.createNodeIndex(
            '{index_name}',
            '{node_label}',
            '{property_name}',
            {dimension},
            'cosine'
        )
        """)

def create_qa_chain(llm, graph):
    template = """
    Task: Generate a Cypher statement to query the graph database.

    Instructions:
    1. Use only relationship types and properties provided in the schema.
    2. If the exact information is not available, try to find related information.
    3. For questions about wars or conflicts, look for FOUGHT_WITH or PARTICIPATED_IN relationships.
    4. Consider both direct and indirect relationships between entities.

    schema:
    {schema}

    Question: {question}

    Cypher Query:
    """ 

    question_prompt = PromptTemplate(
        template=template, 
        input_variables=["schema", "question"] 
    )

    qa_chain = GraphCypherQAChain.from_llm(
        llm=llm,
        graph=graph,
        cypher_prompt=question_prompt,
        verbose=True,
        allow_dangerous_requests=True
    )

    return qa_chain

# # Custom CSS for dark theme
# dark_theme_css = """
# <style>
#     /* Main app */
#     .stApp {
#         background-color: #0E1117;
#         color: #FFFFFF;
#     }
    
#     /* Sidebar */
#     [data-testid="stSidebar"] {
#         background-color: #262730;
#         color: #FFFFFF;
#     }
    
#     /* Inputs */
#     .stTextInput > div > div > input {
#         background-color: #3B3F4B;
#         color: #FFFFFF;
#     }