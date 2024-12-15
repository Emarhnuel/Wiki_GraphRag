# Graphy: Knowledge Graph QA

A Streamlit-based application that creates and queries knowledge graphs using natural language processing and graph database technology.

## Features

- Interactive knowledge graph creation from text input
- Natural language querying of the knowledge graph
- Visual graph representation
- Integration with advanced language models
- Vector-based similarity search
- Neo4j graph database integration

## Prerequisites

- Python 3.x
- Neo4j Database
- OpenAI API key
- Anthropic API key

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows: myenv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a `.env` file in the root directory with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   NEO4J_URI=your_neo4j_uri
   NEO4J_USERNAME=your_username
   NEO4J_PASSWORD=your_password
   ```

## Usage

1. Start the application:
   ```bash
   streamlit run main.py
   ```
2. Use the sidebar to configure your settings
3. Input your text to create a knowledge graph
4. Query the graph using natural language

## Dependencies

- streamlit
- langchain & langchain-community
- neo4j
- transformers
- sentence_transformers
- langchain-anthropic
- langchain-huggingface
- And more (see requirements.txt)

## Project Structure

- `main.py`: Main application file containing the Streamlit interface and core logic
- `utils.py`: Utility functions for knowledge graph creation and querying
- `requirements.txt`: List of Python dependencies
- `.env`: Configuration file for API keys and credentials

## License

This project is licensed under the MIT License.

## Last Updated

2024-12-15
