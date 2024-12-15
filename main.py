import os
import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.graphs import Neo4jGraph
from utils import create_knowledge_graph, create_vector_index, create_qa_chain #dark_theme_css
from neo4j import GraphDatabase
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from openai import OpenAI

client = OpenAI()

def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def main():
    st.set_page_config(
        layout="wide",
        page_title="Graphy: Knowledge Graph QA",
        page_icon=":graph:"
    )
    
    # # Apply dark theme
    # st.markdown()

    st.sidebar.title("Configuration")
    
    with st.sidebar.expander("About"):
        st.markdown("""
        This application creates a knowledge graph from Wikipedia content,
        and answers questions using natural language. It uses LangChain and Anthropic's
        Claude model to generate Cypher queries for the Neo4j database in real-time.
        """)

    # Set Anthropic API key
    anthropic_api_key = st.sidebar.text_input("Enter your Anthropic API Key:", type='password')
    if anthropic_api_key:
        os.environ['ANTHROPIC_API_KEY'] = anthropic_api_key
        st.session_state['ANTHROPIC_API_KEY'] = anthropic_api_key
        st.sidebar.success("Anthropic API Key set successfully.")
        try:
            model_name = "BAAI/bge-large-en-v1.5"
            model_kwargs = {'device': 'cpu'}  # Change to 'cuda' if you have GPU support
            encode_kwargs = {'normalize_embeddings': True}
            embeddings = HuggingFaceBgeEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                query_instruction="Represent this sentence for retrieving relevant articles:"
            )
            llm = ChatAnthropic(model="claude-3-sonnet-20240229", anthropic_api_key=anthropic_api_key)
            st.session_state['embeddings'] = embeddings
            st.session_state['llm'] = llm
        except Exception as e:
            st.error(f"Error initializing components: {str(e)}")

    # Set Neo4j connection details
    st.sidebar.subheader("Connect to Neo4j Database")
    neo4j_url = st.sidebar.text_input("Neo4j URL:", value="neo4j+s://")
    st.sidebar.caption("URL should start with one of: bolt://, bolt+ssc://, bolt+s://, neo4j://, neo4j+ssc://, or neo4j+s://")
    neo4j_username = st.sidebar.text_input("Neo4j Username:", value="neo4j")
    neo4j_password = st.sidebar.text_input("Neo4j Password:", type='password')
    connect_button = st.sidebar.button("Connect")
    
    if connect_button and neo4j_password:
        try:
            # Create a Neo4j driver with a timeout
            driver = GraphDatabase.driver(
                neo4j_url, 
                auth=(neo4j_username, neo4j_password),
                connection_timeout=60  # 60 seconds timeout
            )
            
            # Test the connection
            with driver.session() as session:
                session.run("RETURN 1")
            
            # If successful, create the Neo4jGraph
            graph = Neo4jGraph(
                url=neo4j_url, 
                username=neo4j_username, 
                password=neo4j_password
            )
            
            st.session_state['graph'] = graph
            st.session_state['neo4j_connected'] = True
            st.session_state['neo4j_url'] = neo4j_url
            st.session_state['neo4j_username'] = neo4j_username
            st.session_state['neo4j_password'] = neo4j_password
            st.sidebar.success("Connected to Neo4j database.")
        except Exception as e:
            st.error(f"Failed to connect to Neo4j: {e}")

    st.title("Graphy: Knowledge Graph QA")

    if st.session_state.get('neo4j_connected', False):
        # Topic input
        topic = st.text_input("Enter a topic to create a knowledge graph:")

        if topic and 'qa' not in st.session_state:
            with st.spinner("Creating knowledge graph... This may take a few minutes."):
                try:
                    embeddings = st.session_state.get('embeddings')
                    llm = st.session_state.get('llm')
                    graph = st.session_state.get('graph')
                    
                    if not all([embeddings, llm, graph]):
                        st.error("Please ensure Anthropic API Key is set and Neo4j is connected.")
                    else:
                        docs = create_knowledge_graph(topic, llm, embeddings, graph)
                        
                        # Update the dimension to match your embedding function
                        vector_store = create_vector_index(
                            embeddings,
                            st.session_state['neo4j_url'],
                            st.session_state['neo4j_username'],
                            st.session_state['neo4j_password'],
                            1024
                        )
                        st.session_state['vector_store'] = vector_store

                        st.success(f"Knowledge graph for '{topic}' has been created.")

                        qa = create_qa_chain(llm, graph)
                        st.session_state['qa'] = qa
                        st.session_state['topic'] = topic
                except ImportError as e:
                    st.error(f"Import Error: {str(e)}. Please install the required package.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

        if 'qa' in st.session_state:
            st.subheader("Ask a Question")
            question = st.text_input("Enter your question:")
            if st.button("Submit"):
                with st.spinner("Generating answer..."):
                    try:
                        res = st.session_state['qa'].invoke({"query": question})
                        answer = res['result']
                        
                        if answer.lower().startswith("i don't know") or answer.strip() == "":
                            # Fallback to general QA
                            fallback_template = """
                            Context: The question is about {topic}. The knowledge graph didn't provide a clear answer.
                            Question: {question}
                            Please provide a general answer based on your knowledge:
                            """
                            fallback_prompt = PromptTemplate(template=fallback_template, input_variables=["topic", "question"])
                            fallback_chain = LLMChain(llm=st.session_state['llm'], prompt=fallback_prompt)
                            fallback_response = fallback_chain.run(topic=st.session_state.get('topic', ''), question=question)
                            st.write("\n**Answer (General Knowledge):**\n" + fallback_response)
                        else:
                            st.write("\n**Answer:**\n" + answer)
                    except Exception as e:
                        st.error(f"An error occurred while generating the answer: {str(e)}")
    else:
        st.warning("Please connect to the Neo4j database before creating a knowledge graph.")

if __name__ == "__main__":
    main()