from dotenv import load_dotenv
import os
from llama_index.core import PromptTemplate
from llama_index.core.llms import ChatMessage
from llama_index.llms.lmstudio import LMStudio

# Load environment variables from .env file
load_dotenv(override=True)

# Get configuration from environment/.env
lmstudio_url = os.getenv("LMSTUDIO_URL", "http://localhost:1234")
model_name = os.getenv("MODEL_NAME")

# Initialize LMStudio client
llm = LMStudio(base_url=lmstudio_url, model_name=model_name)

# Define our prompt templates
answer_template = (
    "You are a helpful AI assistant. Your task is to help the user with their query.\n"
    "The user's question is: {query}\n"
    "Please provide a detailed and helpful response."
)

judge_template = (
    "Evaluate if this response adequately answers the original question.\n"
    "Question: {query}\n"
    "Response: {response}\n"
    "Return only 'yes' or 'no' with a confidence score 0-100 in parentheses.\n"
    "Example: 'yes (85)' or 'no (40)'"
)

answer_prompt = PromptTemplate(answer_template)
judge_prompt = PromptTemplate(judge_template)

def main():
    try:
        # Get user input from console
        user_query = input("Enter your prompt: ")
        best_response = None
        best_score = 0
        
        for attempt in range(5):
            # Generate response
            formatted_prompt = answer_prompt.format(query=user_query)
            messages = [ChatMessage(role="user", content=formatted_prompt)]
            response = llm.chat(messages)
            
            # Validate response structure
            if not hasattr(response, 'message') or not hasattr(response.message, 'content'):
                print("Error: Invalid response structure from LM Studio")
                return
                
            response_text = response.message.content
            
            # Judge the response quality
            judge_query = judge_prompt.format(query=user_query, response=response_text)
            judge_messages = [ChatMessage(role="user", content=judge_query)]
            judgement = llm.chat(judge_messages)
            
            if not hasattr(judgement, 'message') or not hasattr(judgement.message, 'content'):
                print("Error: Invalid judgement response")
                continue
                
            judge_text = judgement.message.content.lower()
            
            # Parse judgement
            if 'yes' in judge_text:
                confidence = int(judge_text.split('(')[-1].split(')')[0]) if '(' in judge_text else 80
                if confidence > best_score:
                    best_response = response_text
                    best_score = confidence
                if confidence >= 80:  # Good enough threshold
                    break
            else:
                confidence = int(judge_text.split('(')[-1].split(')')[0]) if '(' in judge_text else 20
                if confidence > best_score:
                    best_response = response_text
                    best_score = confidence
                    
        # Print the best response
        print("\nAssistant Response:")
        print(best_response if best_response else "Error: Could not generate satisfactory response")
        
    except KeyError as e:
        print(f"Validation Error: Missing expected key in response - {e}")
        print("This typically means:")
        print("1. LM Studio isn't running")
        print("2. No model is loaded in LM Studio")
        print("3. Incorrect API URL in .env file")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
