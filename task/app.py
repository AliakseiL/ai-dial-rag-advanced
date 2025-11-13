from task._constants import API_KEY
from task.chat.chat_completion_client import DialChatCompletionClient
from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.embeddings.text_processor import TextProcessor, SearchMode
from task.models.conversation import Conversation
from task.models.message import Message
from task.models.role import Role


# Create system prompt with info that it is RAG powered assistant.
# Explain user message structure (firstly will be provided RAG context and the user question).
# Provide instructions that LLM should use RAG Context when answer on User Question, will restrict LLM to answer
# questions that are not related microwave usage, not related to context or out of history scope
SYSTEM_PROMPT = """You are a RAG-powered assistant that assists users with their questions about microwave usage.
            
## Structure of User message:
`RAG CONTEXT` - Retrieved documents relevant to the query.
`USER QUESTION` - The user's actual question.

## Instructions:
- Use information from `RAG CONTEXT` as context when answering the `USER QUESTION`.
- Cite specific sources when using information from the context.
- Answer ONLY based on conversation history and RAG context.
- If no relevant information exists in `RAG CONTEXT` or conversation history, state that you cannot answer the question.
"""

# Provide structured system prompt, with RAG Context and User Question sections.
USER_PROMPT = """##RAG CONTEXT:
{context}


##USER QUESTION: 
{query}"""


# - create embeddings client with 'text-embedding-3-small-1' model
# - create chat completion client
# - create text processor, DB config: {'host': 'localhost','port': 5433,'database': 'vectordb','user': 'postgres','password': 'postgres'}
# ---
# Create method that will run console chat with such steps:
# - get user input from console
# - retrieve context
# - perform augmentation
# - perform generation
# - it should run in `while` loop (since it is console chat)

class MicrowaveRAG:
    def __init__(self):
        self.embeddings_client = DialEmbeddingsClient(
            deployment_name='text-embedding-3-small-1',
            api_key=API_KEY
        )
        self.chat_client = DialChatCompletionClient(
            deployment_name='gpt-4o',
            api_key=API_KEY
        )
        db_config = {
            'host': 'localhost',
            'port': 5433,
            'database': 'vectordb',
            'user': 'postgres',
            'password': 'postgres'
        }
        self.text_processor = TextProcessor(self.embeddings_client, db_config)

    def run_console_chat(self):
        print("Welcome to the Microwave RAG Assistant! Type 'exit' to quit.")
        load_context = input("\nLoad context to VectorDB (y/n)? > ").strip()
        if load_context.lower().strip() == 'y':
            self.text_processor.process_text_file(
                file_name='embeddings/microwave_manual.txt',
                chunk_size=400,
                overlap=40,
                dimensions=1536,
                truncate_table=True
            )
            print("=" * 100)


        conversation = Conversation()
        conversation.add_message(Message(role=Role.SYSTEM, content=SYSTEM_PROMPT))

        while True:
            user_input = input(">")
            if user_input.lower() == 'exit':
                print("Exiting the chat. Goodbye!")
                break

            # Retrieve context
            context_chunks = self.text_processor.search(
                search_mode=SearchMode.EUCLIDIAN_DISTANCE,
                user_request=user_input,
                top_k=5,
                score_threshold=0.01,
                dimensions=1536
            )
            context = "\n\n".join(context_chunks)

            # Augment user input with context
            augmented_user_input = USER_PROMPT.format(context=context, query=user_input)
            print(f"Augmented user input:\n{augmented_user_input}")
            conversation.add_message(Message(role=Role.USER, content=augmented_user_input))

            # Generate response
            response = self.chat_client.get_completion(conversation.get_messages(), True)
            conversation.add_message(Message(role=Role.AI, content=response))

            print(f"Assistant: {response.content}")

if __name__ == "__main__":
    MicrowaveRAG.run_console_chat(self=MicrowaveRAG())

# TODO:
#  PAY ATTENTION THAT YOU NEED TO RUN Postgres DB ON THE 5433 WITH PGVECTOR EXTENSION!
#  RUN docker-compose.yml