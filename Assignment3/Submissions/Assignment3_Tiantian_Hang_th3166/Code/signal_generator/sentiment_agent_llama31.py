"""
Sentiment Agent - Uses Llama 3.1 Base Model (no LoRA)
Updated for Llama 3.1 support with chat template
"""

import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to load from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from news_fetcher.prompt_builder import PromptBuilder
from news_fetcher.data_fetcher import NewsDataFetcher


class SentimentAgent:
    """
    Sentiment analysis agent using Llama 3.1 Base Model (no LoRA)
    """
    
    def __init__(self, model_path=None, tokenizer_path=None, hf_token=None):
        """
        Initialize sentiment agent with Llama 3.1
        
        Args:
            model_path: Path to base model (default: 'meta-llama/Llama-3.1-8B-Instruct')
            tokenizer_path: Path to tokenizer (default: same as model_path)
            hf_token: HuggingFace token (if None, uses HF_TOKEN env var)
        """
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HuggingFace token is required. Set HF_TOKEN environment variable.")
        
        # Default to Llama 3.1-8B-Instruct
        base_model_path = model_path or 'meta-llama/Llama-3.1-8B-Instruct'
        tokenizer_path = tokenizer_path or base_model_path
        
        print(f"Loading Llama 3.1 model: {base_model_path}")
        print("Note: Using Base Model (no LoRA adapter)")
        
        self.base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            token=self.hf_token,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        # Use base model directly (no LoRA)
        self.model = self.base_model
        self.model = self.model.eval()
        
        print(f"Loading tokenizer: {tokenizer_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            token=self.hf_token
        )
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize prompt builder
        self.prompt_builder = PromptBuilder()
        self.data_fetcher = NewsDataFetcher()
        
        # System prompt (same as before)
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
    
    def format_prompt(self, system_prompt, user_prompt):
        """
        Format prompt using Llama 3.1 chat template
        
        Args:
            system_prompt: System instruction
            user_prompt: User input prompt
            
        Returns:
            Formatted prompt string
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        try:
            # Use tokenizer's chat template (recommended for Llama 3.1)
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted
        except Exception as e:
            print(f"⚠️  Chat template not available, using manual format: {e}")
            # Fallback to manual format
            return (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
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
        
        # Format prompt using Llama 3.1 chat template
        formatted_prompt = self.format_prompt(self.SYSTEM_PROMPT, prompt)
        
        # Generate prediction
        inputs = self.tokenizer(formatted_prompt, return_tensors='pt')
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        
        with torch.no_grad():
            res = self.model.generate(
                **inputs,
                max_length=4096,
                do_sample=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                temperature=0.7,
                top_p=0.9
            )
        
        output = self.tokenizer.decode(res[0], skip_special_tokens=True)
        
        # Extract model-generated content (remove prompt)
        # For Llama 3.1, model output starts after [/INST] or the last assistant marker
        model_output = self._extract_model_output(output, formatted_prompt)
        
        # Debug: Log if extraction seems to have failed
        if len(model_output) < 50 or '[Positive Developments]' not in model_output:
            print(f"⚠️  Warning: Model output extraction may have failed for {symbol}")
            print(f"   Model output length: {len(model_output)}")
            print(f"   First 200 chars: {model_output[:200]}")
        
        # Parse output using model-generated content
        result = self._parse_output(model_output, symbol)
        result['raw_output'] = model_output  # Save only model-generated content
        
        # Debug: Log if prediction parsing failed
        if result['predicted_return'] == 0.0 and '[Prediction' in model_output:
            print(f"⚠️  Warning: Could not parse prediction for {symbol}")
            print(f"   Looking for prediction pattern in output...")
            # Try to find any prediction-like text
            prediction_like = re.search(r'Prediction[:\s]+.*?(?:\n|$)', model_output, re.IGNORECASE)
            if prediction_like:
                print(f"   Found: {prediction_like.group(0)[:100]}")
        
        return result
    
    def _extract_model_output(self, full_output, prompt):
        """
        Extract model-generated content from full output (remove prompt)
        
        Args:
            full_output: Full output including prompt and model response
            prompt: The original prompt that was sent to the model
            
        Returns:
            str: Model-generated content only
        """
        # Try to find the model output by looking for common markers
        # Method 1: Look for [/INST] marker (Llama 3.1 format)
        inst_marker = '[/INST]'
        if inst_marker in full_output:
            parts = full_output.split(inst_marker, 1)
            if len(parts) > 1:
                model_output = parts[1].strip()
                # Remove any trailing special tokens
                model_output = re.sub(r'<\|.*?\|>', '', model_output).strip()
                return model_output
        
        # Method 2: Look for assistant header marker
        assistant_marker = '<|start_header_id|>assistant<|end_header_id|>'
        if assistant_marker in full_output:
            parts = full_output.split(assistant_marker, 1)
            if len(parts) > 1:
                model_output = parts[1].strip()
                # Remove any trailing special tokens
                model_output = re.sub(r'<\|.*?\|>', '', model_output).strip()
                return model_output
        
        # Method 3: Try to find where prompt ends
        # Find the last occurrence of a distinctive part of the prompt
        prompt_end_markers = [
            'Provide a summary analysis to support your prediction.',
            'Then make your prediction',
            'stock price movement for next week'
        ]
        
        for marker in prompt_end_markers:
            if marker in full_output:
                idx = full_output.rfind(marker)
                if idx > 0:
                    # Get text after the marker
                    potential_output = full_output[idx + len(marker):].strip()
                    # Check if it starts with [Positive Developments] or similar
                    if '[Positive Developments]' in potential_output or 'Prediction:' in potential_output:
                        # Remove any leading special tokens or markers
                        potential_output = re.sub(r'^\[INST\]\s*', '', potential_output)
                        potential_output = re.sub(r'<\|.*?\|>', '', potential_output).strip()
                        return potential_output
        
        # Fallback: If we can't find a clear separator, try to remove the prompt
        # by finding where the prompt text ends
        if prompt in full_output:
            idx = full_output.find(prompt)
            if idx >= 0:
                model_output = full_output[idx + len(prompt):].strip()
                # Remove any leading special tokens
                model_output = re.sub(r'^\[INST\]\s*', '', model_output)
                model_output = re.sub(r'<\|.*?\|>', '', model_output).strip()
                return model_output
        
        # Last resort: return full output (shouldn't happen in normal cases)
        return full_output
    
    def _parse_output(self, output, symbol):
        """
        Parse model output to extract signal
        
        Args:
            output: Model-generated output (should not include prompt)
            symbol: Stock ticker symbol (for context)
            
        Returns:
            dict with signal, confidence, etc.
        """
        # Extract prediction - support multiple formats
        # Examples: 
        # - "Prediction: Up by 3%"
        # - "Prediction: Down by 1-2%"
        # - "likely to decrease by 2-5%"
        # - "Stable to Slightly Positive Movement"
        predicted_return = 0.0
        
        # Pattern 1: Standard format "Prediction: Up/Down by X%"
        prediction_match = re.search(
            r'Prediction:.*?(?:up|down)\s*(?:by)?\s*([\d.]+)(?:-([\d.]+))?%',
            output,
            re.IGNORECASE
        )
        
        if prediction_match:
            if prediction_match.group(2):  # Range format (e.g., "3-4%")
                value1 = float(prediction_match.group(1))
                value2 = float(prediction_match.group(2))
                value = (value1 + value2) / 2
            else:  # Single value format (e.g., "3%")
                value = float(prediction_match.group(1))
            
            # Check direction from the matched text
            prediction_text = prediction_match.group(0).lower()
            if 'down' in prediction_text or 'decrease' in prediction_text or 'decline' in prediction_text:
                predicted_return = -value / 100
            elif 'up' in prediction_text or 'increase' in prediction_text or 'rise' in prediction_text:
                predicted_return = value / 100
            else:
                predicted_return = value / 100
        else:
            # Pattern 2: "likely to decrease/increase by X%"
            alt_match = re.search(
                r'likely to (?:decrease|decline|down|increase|rise|up).*?by\s*([\d.]+)(?:-([\d.]+))?%',
                output,
                re.IGNORECASE
            )
            if alt_match:
                if alt_match.group(2):  # Range format
                    value1 = float(alt_match.group(1))
                    value2 = float(alt_match.group(2))
                    value = (value1 + value2) / 2
                else:
                    value = float(alt_match.group(1))
                
                prediction_text = alt_match.group(0).lower()
                if 'down' in prediction_text or 'decrease' in prediction_text or 'decline' in prediction_text:
                    predicted_return = -value / 100
                elif 'up' in prediction_text or 'increase' in prediction_text or 'rise' in prediction_text:
                    predicted_return = value / 100
            else:
                # Pattern 3: "decrease/increase by X%" (without "likely to")
                alt_match2 = re.search(
                    r'(?:decrease|decline|down|increase|rise|up).*?by\s*([\d.]+)(?:-([\d.]+))?%',
                    output,
                    re.IGNORECASE
                )
                if alt_match2:
                    if alt_match2.group(2):  # Range format
                        value1 = float(alt_match2.group(1))
                        value2 = float(alt_match2.group(2))
                        value = (value1 + value2) / 2
                    else:
                        value = float(alt_match2.group(1))
                    
                    prediction_text = alt_match2.group(0).lower()
                    if 'down' in prediction_text or 'decrease' in prediction_text or 'decline' in prediction_text:
                        predicted_return = -value / 100
                    elif 'up' in prediction_text or 'increase' in prediction_text or 'rise' in prediction_text:
                        predicted_return = value / 100
                else:
                    # Pattern 4: Descriptive predictions like "Stable to Slightly Positive Movement"
                    prediction_lower = output.lower()
                    if 'stable' in prediction_lower or 'neutral' in prediction_lower:
                        predicted_return = 0.0
                    elif 'slightly positive' in prediction_lower or 'slight increase' in prediction_lower:
                        predicted_return = 0.01  # 1% small positive
                    elif 'slightly negative' in prediction_lower or 'slight decrease' in prediction_lower:
                        predicted_return = -0.01  # 1% small negative
                    elif 'positive' in prediction_lower and 'movement' in prediction_lower:
                        predicted_return = 0.02  # 2% positive (default for positive movement)
                    elif 'negative' in prediction_lower and 'movement' in prediction_lower:
                        predicted_return = -0.02  # 2% negative (default for negative movement)
        
        # Count positive and negative factors - count actual list items, not section headers
        # Support multiple formats: [Positive Developments]:, **Positive Developments:**, Positive Developments:
        positive_count = 0
        negative_count = 0
        
        # Extract Positive Developments section - support multiple formats
        positive_patterns = [
            r'\[Positive Developments\]:\s*(.*?)(?=\[Potential Concerns\]:|\[Prediction|$)',
            r'\*\*Positive Developments\*\*:\s*(.*?)(?=\*\*Potential Concerns\*\*:|\*\*Prediction|$)',
            r'Positive Developments:\s*(.*?)(?=Potential Concerns:|Prediction|$)'
        ]
        
        positive_section = None
        for pattern in positive_patterns:
            positive_section_match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
            if positive_section_match:
                positive_section = positive_section_match.group(1)
                break
        
        if positive_section:
            # Count numbered items (1., 2., 3., etc.)
            positive_items = re.findall(r'^\d+\.', positive_section, re.MULTILINE)
            positive_count = len(positive_items)
        
        # Extract Potential Concerns section - support multiple formats
        concern_patterns = [
            r'\[Potential Concerns\]:\s*(.*?)(?=\[Prediction|$)',
            r'\*\*Potential Concerns\*\*:\s*(.*?)(?=\*\*Prediction|$)',
            r'Potential Concerns:\s*(.*?)(?=Prediction|$)'
        ]
        
        concern_section = None
        for pattern in concern_patterns:
            concern_section_match = re.search(pattern, output, re.IGNORECASE | re.DOTALL)
            if concern_section_match:
                concern_section = concern_section_match.group(1)
                break
        
        if concern_section:
            # Count numbered items (1., 2., 3., etc.)
            concern_items = re.findall(r'^\d+\.', concern_section, re.MULTILINE)
            negative_count = len(concern_items)
        
        # Fallback: if section extraction failed, count section headers
        if positive_count == 0:
            positive_patterns_fallback = [
                r'\[Positive Developments\]:',
                r'\*\*Positive Developments\*\*:',
                r'Positive Developments:'
            ]
            for pattern in positive_patterns_fallback:
                positive_matches = re.findall(pattern, output, re.IGNORECASE)
                if positive_matches:
                    positive_count = len(positive_matches)
                    break
        
        if negative_count == 0:
            concern_patterns_fallback = [
                r'\[Potential Concerns\]:',
                r'\*\*Potential Concerns\*\*:',
                r'Potential Concerns:'
            ]
            for pattern in concern_patterns_fallback:
                concern_matches = re.findall(pattern, output, re.IGNORECASE)
                if concern_matches:
                    negative_count = len(concern_matches)
                    break
        
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
            'negative_count': negative_count,
            'predicted_return': predicted_return
        }
