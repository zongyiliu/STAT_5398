"""
Sentiment Agent - Uses FinGPT to analyze news and generate trading signals
Adapted from FinGPT_Forecaster
"""

import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to load from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use environment variables directly

from news_fetcher.prompt_builder import PromptBuilder
from news_fetcher.data_fetcher import NewsDataFetcher


class SentimentAgent:
    """
    Sentiment analysis agent using FinGPT model
    """
    
    def __init__(self, model_path=None, tokenizer_path=None, hf_token=None):
        """
        Initialize sentiment agent
        
        Args:
            model_path: Path to base model (default: 'meta-llama/Llama-2-7b-chat-hf')
            tokenizer_path: Path to tokenizer (default: same as model_path)
            hf_token: HuggingFace token (if None, uses HF_TOKEN env var)
        """
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HuggingFace token is required. Set HF_TOKEN environment variable.")
        
        # Default model paths
        base_model_path = model_path or 'meta-llama/Llama-2-7b-chat-hf'
        lora_model_path = 'FinGPT/fingpt-forecaster_dow30_llama2-7b_lora'
        tokenizer_path = tokenizer_path or base_model_path
        
        # Detect device
        if torch.cuda.is_available():
            device = "cuda"
            device_map = "auto"
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"✓ GPU detected: {gpu_name}")
            print(f"  GPU Memory: {gpu_memory:.2f} GB")
        else:
            device = "cpu"
            device_map = "cpu"  # Use CPU explicitly to avoid meta device issues
            print("⚠️  No GPU detected. Using CPU (this will be VERY slow)")
            print("")
            print("   IMPORTANT: For Colab users:")
            print("   1. Go to: Runtime → Change runtime type")
            print("   2. Select: Hardware accelerator → GPU")
            print("   3. Choose: T4 (free) or A100 (paid)")
            print("   4. Click: Save")
            print("   5. Restart runtime and run again")
            print("")
            print("   ⚠️  Warning: CPU inference will take 20+ minutes per stock!")
            print("   ⚠️  Model loading alone may take 10+ minutes on CPU!")
            print("")
            # Ask for confirmation in interactive mode
            try:
                import sys
                if sys.stdin.isatty():  # Interactive mode
                    response = input("   Continue with CPU? (y/N): ").strip().lower()
                    if response != 'y':
                        raise RuntimeError("Aborted: GPU is required for reasonable performance")
            except:
                pass  # Non-interactive mode, continue anyway
        
        print(f"Loading base model: {base_model_path}")
        print("   This may take several minutes, especially on CPU...")
        print("   Model size: ~13GB, download may take 10-20 minutes")
        print("   Please be patient and do not close the browser tab...")
        
        # Retry mechanism for model loading
        max_retries = 3
        retry_delay = 60  # Wait 60 seconds between retries
        
        for attempt in range(max_retries):
            try:
                print(f"   Attempt {attempt + 1}/{max_retries}...")
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    token=self.hf_token,
                    trust_remote_code=True,
                    device_map=device_map,
                    dtype=torch.float16 if device == "cuda" else torch.float32,
                    resume_download=True,  # Resume interrupted downloads
                    local_files_only=False,  # Allow downloading
                )
                print("   ✓ Base model loaded successfully")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"   ⚠️  Attempt {attempt + 1} failed: {e}")
                    print(f"   Retrying in {retry_delay} seconds...")
                    import time
                    time.sleep(retry_delay)
                else:
                    print(f"   ✗ Failed to load model after {max_retries} attempts")
                    raise
        
        # Ensure model is on the correct device (not meta)
        if device_map == "cpu":
            # If using CPU, ensure all parameters are on CPU
            self.base_model = self.base_model.to(device)
        
        # Try to load LoRA adapter, fallback to base model if it fails
        print(f"Loading LoRA adapter: {lora_model_path}")
        try:
            self.model = PeftModel.from_pretrained(
                self.base_model,
                lora_model_path,
            )
            print("✓ LoRA adapter loaded successfully")
        except Exception as e:
            print(f"⚠️  Failed to load LoRA adapter: {e}")
            print("   Using base model without LoRA (this may affect performance)")
            self.model = self.base_model
        
        # Ensure model is on the correct device (not meta)
        if device_map == "cpu":
            self.model = self.model.to(device)
        
        self.model = self.model.eval()
        
        # Store device for later use
        self.device = device
        
        print(f"Loading tokenizer: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            token=self.hf_token
        )
        
        # Initialize prompt builder
        self.prompt_builder = PromptBuilder()
        self.data_fetcher = NewsDataFetcher()
        
        # Prompt format constants
        self.B_INST, self.E_INST = "[INST]", "[/INST]"
        self.B_SYS, self.E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        
        self.SYSTEM_PROMPT = (
            "You are a seasoned stock market analyst. Your task is to list the positive "
            "developments and potential concerns for companies based on relevant news and "
            "basic financials from the past weeks, then provide an analysis and prediction "
            "for the companies' stock price movement for the upcoming week. "
            "Your answer format should be as follows:\n\n"
            "[Positive Developments]:\n1. ...\n\n"
            "[Potential Concerns]:\n1. ...\n\n"
            "[Prediction & Analysis]:\nPrediction: ...\nAnalysis: ..."
        )
    
    def analyze(self, news_data, symbol, date, n_weeks=3):
        """
        Analyze news sentiment and generate trading signal
        
        Args:
            news_data: DataFrame with news data (optional, if None will fetch)
            symbol: Stock ticker symbol
            date: Trading date string
            n_weeks: Number of weeks to look back
            
        Returns:
            dict: {
                'signal': 1 (buy), -1 (sell), 0 (hold),
                'confidence': 0.0-1.0,
                'positive_count': int,
                'negative_count': int,
                'predicted_return': float,
                'raw_output': str
            }
        """
        # Fetch data if not provided
        if news_data is None:
            news_data = self.data_fetcher.fetch_all_data(symbol, date, n_weeks)
        
        # Build prompt
        info, prompt = self.prompt_builder.get_all_prompts_online(
            symbol, news_data, date, with_basics=True
        )
        
        # Format prompt for Llama
        formatted_prompt = (
            self.B_INST + self.B_SYS + self.SYSTEM_PROMPT + self.E_SYS + 
            prompt + self.E_INST
        )
        
        # Generate prediction
        inputs = self.tokenizer(formatted_prompt, return_tensors='pt')
        # Move inputs to the correct device
        # Handle models with device_map (may have multiple devices)
        if hasattr(self.model, 'device'):
            target_device = self.model.device
        elif hasattr(self.model, 'hf_device_map'):
            # If using device_map, find the first non-meta device
            device_map = self.model.hf_device_map
            target_device = None
            for layer_device in device_map.values():
                if layer_device != 'meta' and layer_device is not None:
                    target_device = layer_device
                    break
            if target_device is None:
                target_device = self.device
        else:
            target_device = self.device
        
        inputs = {key: value.to(target_device) for key, value in inputs.items()}
        
        with torch.no_grad():
            res = self.model.generate(
                **inputs,
                max_length=4096,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                temperature=0.7,
                top_p=0.9
            )
        
        output = self.tokenizer.decode(res[0], skip_special_tokens=True)
        
        # Parse output
        result = self._parse_output(output, symbol)
        result['raw_output'] = output
        
        return result
    
    def _parse_output(self, output, symbol):
        """
        Parse FinGPT output to extract signal
        
        Args:
            output: Raw model output
            symbol: Stock ticker symbol (for context)
            
        Returns:
            dict with signal, confidence, etc.
        """
        # Extract prediction - support both single value and range formats
        # Examples: "Up by 3%", "Up by 3-4%", "Down by 2%", "Down by 2-3%"
        prediction_match = re.search(
            r'Prediction:\s*(?:up|down)\s*(?:by)?\s*([\d.]+)(?:-([\d.]+))?%',
            output,
            re.IGNORECASE
        )
        
        predicted_return = 0.0
        if prediction_match:
            if prediction_match.group(2):  # Range format (e.g., "3-4%")
                value1 = float(prediction_match.group(1))
                value2 = float(prediction_match.group(2))
                # Use average of the range
                value = (value1 + value2) / 2
            else:  # Single value format (e.g., "3%")
                value = float(prediction_match.group(1))
            
            # Check if up or down
            # Look for "up" or "down" near the prediction
            prediction_text = prediction_match.group(0).lower()
            if 'up' in prediction_text or 'increase' in output.lower():
                predicted_return = value / 100
            elif 'down' in prediction_text or 'decrease' in output.lower():
                predicted_return = -value / 100
            else:
                # Default to positive if unclear
                predicted_return = value / 100
        
        # Count positive and negative factors
        positive_matches = re.findall(r'\[Positive Developments\]:', output, re.IGNORECASE)
        concern_matches = re.findall(r'\[Potential Concerns\]:', output, re.IGNORECASE)
        
        positive_count = len(positive_matches)
        concern_count = len(concern_matches)
        
        # Generate signal based on prediction
        if predicted_return > 0.02:  # Predict > 2% increase
            signal = 1
            confidence = min(abs(predicted_return) * 10, 1.0)
        elif predicted_return < -0.02:  # Predict > 2% decrease
            signal = -1
            confidence = min(abs(predicted_return) * 10, 1.0)
        else:
            signal = 0
            confidence = 0.5
        
        return {
            'signal': signal,
            'confidence': confidence,
            'positive_count': positive_count,
            'negative_count': concern_count,
            'predicted_return': predicted_return
        }

