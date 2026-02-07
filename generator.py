"""
Generator Module
LLM-based response generation with grounded prompts and citations
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    OPENAI_API_KEY, OPENAI_MODEL, 
    GOOGLE_API_KEY, GOOGLE_MODEL,
    DEFAULT_LLM_PROVIDER,
    TEMPERATURE, MAX_TOKENS,
    USE_LOCAL_LLM, OLLAMA_MODEL
)


# Prompt template for grounded responses
SYSTEM_PROMPT = """You are a professional AI assistant providing concise, grounded answers from context documents.

RULES:
1. DIRECTNESS: Answer the question directly. Do not use filler phrases.
2. CONCISENESS: Provide ONLY the requested information. Remove unnecessary details.
3. CITATIONS: Use [1], [2] at the end of specific sentences. 
   - DO NOT cite multiple sources [1, 2, 3, 4] unless EVERY cited source contains unique information for THAT specific sentence.
   - If a fact is found in multiple sources, cite only the most relevant one or two.
4. MISSING INFO: If the context is insufficient, state: "The provided documents do not contain enough information to answer this question."
5. NO METADATA: Never output JSON, system signatures, or internal metadata.

FORMAT:
- Use bullet points ONLY for lists.
- Use bold text for key terms.
- Citation format: [1], [1, 2]."""

USER_PROMPT_TEMPLATE = """Context:
{context}

Question: {question}

Answer concisely with citations [1], [2] as specified in the rules:"""


class Generator:
    """
    LLM-based response generator with citation support.
    Supports OpenAI models and local LLMs via Ollama.
    """
    
    def __init__(
        self,
        model: str = None,
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKENS
    ):
        """
        Initialize the generator.
        
        Args:
            model: Model name (overrides config)
            temperature: Response temperature (0-1)
            max_tokens: Maximum tokens in response
        """
        self.use_local = USE_LOCAL_LLM or DEFAULT_LLM_PROVIDER == "local"
        
        if self.use_local:
            self.provider = "local"
            self.model = model or OLLAMA_MODEL
        elif DEFAULT_LLM_PROVIDER == "google" or GOOGLE_API_KEY:
            self.provider = "google"
            self.model = model or GOOGLE_MODEL
        else:
            self.provider = "openai"
            self.model = model or OPENAI_MODEL
            
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm = None
    
    @property
    def llm(self):
        """Lazy load the LLM."""
        if self._llm is None:
            if self.provider == "local":
                self._init_local_llm()
            elif self.provider == "google":
                self._init_google_llm()
            else:
                self._init_openai_llm()
        return self._llm
    
    def _init_google_llm(self):
        """Initialize Google Gemini LLM."""
        if not GOOGLE_API_KEY:
            raise ValueError("Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
        
        print(f"[*] Initializing Google model: {self.model}")
        from langchain_google_genai import ChatGoogleGenerativeAI
        
        self._llm = ChatGoogleGenerativeAI(
            model=self.model,
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            google_api_key=GOOGLE_API_KEY
        )
        print(f"[OK] Google model initialized")
    
    def _init_openai_llm(self):
        """Initialize OpenAI LLM."""
        if not OPENAI_API_KEY:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file "
                "or set USE_LOCAL_LLM=true to use a local model."
            )
        
        print(f"[*] Initializing OpenAI model: {self.model}")
        from langchain_openai import ChatOpenAI
        
        self._llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=OPENAI_API_KEY
        )
        print(f"[OK] OpenAI model initialized")
    
    def _init_local_llm(self):
        """Initialize local LLM via Ollama."""
        print(f"[*] Initializing local model: {self.model}")
        try:
            from langchain_community.llms import Ollama
            
            self._llm = Ollama(
                model=self.model,
                temperature=self.temperature
            )
            print(f"[OK] Local model initialized")
        except Exception as e:
            raise ValueError(
                f"Failed to initialize local LLM. Make sure Ollama is running. Error: {e}"
            )
    
    def generate(
        self,
        question: str,
        context: str,
        sources: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Generate a response based on context.
        
        Args:
            question: User's question
            context: Retrieved context documents
            sources: List of source citations
            
        Returns:
            Dictionary with answer and citations
        """
        if not context:
            return {
                "answer": "I don't have any documents ingested yet. Please upload and ingest some documents first, then ask your question again.",
                "citations": [],
                "has_context": False
            }
        
        # Format the prompt
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )
        
        # Generate response
        print(f"[*] Generating response...")
        try:
            if hasattr(self.llm, 'invoke'):
                # LangChain v0.1+ API
                from langchain_core.messages import HumanMessage, SystemMessage
                messages = [
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_prompt)
                ]
                response = self.llm.invoke(messages)
                
                # Handle different response formats (Gemini can return a list of parts)
                if hasattr(response, 'content'):
                    content = response.content
                    if isinstance(content, list):
                        texts = []
                        for part in content:
                            if isinstance(part, dict) and 'text' in part:
                                texts.append(part['text'])
                            elif isinstance(part, str):
                                texts.append(part)
                            else:
                                texts.append(str(part))
                        answer = "".join(texts)
                    else:
                        answer = str(content)
                else:
                    answer = str(response)
            else:
                # Fallback for older API
                response = self.llm(user_prompt)
                answer = str(response)
            
            print(f"[OK] Response generated")
            
            return {
                "answer": answer,
                "citations": sources or [],
                "has_context": True
            }
            
        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR] Error generating response: {error_msg}")
            
            # Provide helpful error messages
            if "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                return {
                    "answer": "[WARNING] API key error. Please check your OpenAI API key in the .env file.",
                    "citations": [],
                    "has_context": False,
                    "error": error_msg
                }
            
            return {
                "answer": f"[WARNING] Error generating response: {error_msg}",
                "citations": [],
                "has_context": False,
                "error": error_msg
            }
    
    def generate_streaming(
        self,
        question: str,
        context: str
    ):
        """
        Generate a streaming response (for real-time display).
        
        Yields:
            Response chunks as they're generated
        """
        if not context:
            yield "I don't have any documents ingested yet. Please upload and ingest some documents first."
            return
        
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )
        
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]
            
            for chunk in self.llm.stream(messages):
                content = getattr(chunk, 'content', str(chunk))
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and 'text' in part:
                            yield part['text']
                        elif isinstance(part, str):
                            yield part
                else:
                    yield str(content)
                    
        except Exception as e:
            yield f"[WARNING] Error: {e}"


# Singleton instance
_generator = None


def get_generator() -> Generator:
    """Get the singleton generator instance."""
    global _generator
    if _generator is None:
        _generator = Generator()
    return _generator


if __name__ == "__main__":
    # Test generator (requires API key)
    generator = Generator()
    
    test_context = """[1] Source: test.txt
Machine learning is a subset of artificial intelligence that enables systems to learn from data.

[2] Source: test.txt
Deep learning uses neural networks with multiple layers to process complex patterns."""
    
    result = generator.generate(
        question="What is machine learning?",
        context=test_context,
        sources=[{"citation": 1, "source": "test.txt"}]
    )
    
    print("\n[*] Generated Answer:")
    print(result["answer"])
    print("\n[*] Citations:")
    for cite in result["citations"]:
        print(f"  [{cite['citation']}] {cite['source']}")
