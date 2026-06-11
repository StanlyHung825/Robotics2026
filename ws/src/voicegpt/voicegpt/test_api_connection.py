import os
import sys
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

def test_github_token():
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    # Remove any non-ASCII or hidden unicode characters
    token = "".join(c for c in token if ord(c) < 128)
    
    if not token:
        print("ERROR: GITHUB_TOKEN is not set in environment.")
        sys.exit(1)
        
    print(f"Token (first 10 chars): {token[:10]}...")
    endpoint = "https://models.github.ai/inference"
    model = "gpt-4o-mini"
    
    try:
        client = ChatCompletionsClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(token),
        )
        print("Calling GitHub Models API (gpt-4o)...")
        response = client.complete(
            messages=[
                SystemMessage("Output exactly 'OK' and nothing else."),
                UserMessage("Hello"),
            ],
            model=model,
        )
        reply = response.choices[0].message.content.strip()
        print(f"SUCCESS! GPT responded: {reply}")
    except Exception as e:
        print(f"\nAPI CALL FAILED!")
        print(f"Error details: {e}")
        if "401" in str(e) or "Unauthorized" in str(e):
            print("Explanation: The GITHUB_TOKEN is invalid (401 Unauthorized). Check if it is expired or has typos.")
        elif "429" in str(e) or "Rate limit" in str(e) or "quota" in str(e):
            print("Explanation: You have run out of GitHub Models API rate limit or tokens (429 Rate Limit exceeded).")
        else:
            print("Explanation: This might be a network error, DNS resolution failure, or proxy issue.")

if __name__ == "__main__":
    test_github_token()
