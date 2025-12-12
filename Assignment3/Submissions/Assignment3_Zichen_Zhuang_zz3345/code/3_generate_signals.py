import pandas as pd
import sys
import json
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent))
import config

def load_fingpt_model():
    print("Loading FinGPT Model")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel
    
    # Determine which model to use
    if config.MODEL_TYPE == "base":
        model_path = config.FINGPT_BASE_MODEL
        use_peft = False
        print(f"Using base model: {model_path}")
    elif config.MODEL_TYPE == "local_finetuned":
        base_model_path = config.FINGPT_BASE_MODEL
        adapter_path = config.FINGPT_FINETUNED_LOCAL
        use_peft = True
        print(f"Using local fine-tuned model")
        print(f"Base: {base_model_path}")
        print(f"Adapter: {adapter_path}")
    elif config.MODEL_TYPE == "colab_finetuned":
        base_model_path = config.FINGPT_BASE_MODEL
        adapter_path = config.FINGPT_FINETUNED_COLAB
        use_peft = True
        print(f"Using Colab fine-tuned model")
        print(f"Base: {base_model_path}")
        print(f"Adapter: {adapter_path}")
    else:
        raise ValueError(f"Unknown model type: {config.MODEL_TYPE}")
    
    # Check if model paths exist
    model_exists = False
    if use_peft:
        model_exists = Path(base_model_path).exists() and Path(adapter_path).exists()
    else:
        model_exists = Path(model_path).exists()
    
    if not model_exists:
        return None, None, True  # Return None model with simulation flag
    
    # Load model with 8-bit quantization
    print("\nLoading model with 8-bit")
    
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.float16
    )
    
    if use_peft:
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        
        # Load adapter
        model = PeftModel.from_pretrained(
            base_model,
            adapter_path,
            local_files_only=True
        )
        print("Loaded base model + LoRA adapter")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        print("Loaded base model")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path if use_peft else model_path,
        trust_remote_code=True,
        local_files_only=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model.eval()
    print("Model loaded successfully!\n")
    
    return model, tokenizer, False  # Return with simulation=False

def create_prompt_from_news(ticker, news_items, company_info=""):
    prompt = f"""Analyze the following news articles about {ticker} and predict the stock price movement.

Company: {ticker}
{company_info}

Recent News:
"""
    
    for i, news in enumerate(news_items, 1):
        prompt += f"\n{i}. {news['headline']}\n   {news['summary']}\n"
    
    prompt += """
Based on the above information, provide:
1. [Positive Developments]: Key positive factors
2. [Potential Concerns]: Key risks and concerns  
3. [Prediction & Analysis]: Predict price direction (up/down) and estimated percentage change

Format your response as:
[Positive Developments]:
<your analysis>

[Potential Concerns]:
<your analysis>

[Prediction & Analysis]:
Direction: <up/down>
Estimated Change: <percentage>
Reasoning: <your reasoning>
"""
    
    return prompt

def simulate_model_prediction(news_items):
    # Count positive vs negative sentiment
    positive_count = sum(1 for n in news_items if n.get('sentiment') == 'positive')
    negative_count = sum(1 for n in news_items if n.get('sentiment') == 'negative')
    
    if positive_count > negative_count:
        direction = "up"
        confidence = positive_count / len(news_items)
        estimated_change = confidence * 5.0  # Up to 5% change
    elif negative_count > positive_count:
        direction = "down"
        confidence = negative_count / len(news_items)
        estimated_change = -confidence * 5.0  # Down to -5% change
    else:
        direction = "neutral"
        estimated_change = 0.0
    
    return {
        'direction': direction,
        'estimated_change': estimated_change,
        'confidence': (positive_count - negative_count) / len(news_items) if news_items else 0
    }

def generate_signals():
    # Load news data
    news_file = config.NEWS_DIR / "collected_news.json"
    if not news_file.exists():
        print(f"ERROR: News data not found: {news_file}")
        print("Please run 2 first")
        return False
    
    with open(news_file, 'r') as f:
        news_data = json.load(f)
    
    news_df = pd.DataFrame(news_data)
    print(f"Loaded {len(news_df)} news articles for {news_df['ticker'].nunique()} stocks")
    
    # Load model
    model, tokenizer, simulation_mode = load_fingpt_model()
    
    if simulation_mode:
        print("Running in SIMULATION MODE (model not available)")
    
    # Group news by stock and period
    news_df['date'] = pd.to_datetime(news_df['date'])
    news_df['period'] = news_df['date'].dt.to_period('M')  # Monthly periods
    
    # Generate signals for each stock-period
    signals = []
    
    grouped = news_df.groupby(['ticker', 'gvkey', 'period'])
    
    print(f"\nGenerating signals for {len(grouped)} stock-period combinations...")
    
    for (ticker, gvkey, period), group in tqdm(grouped, desc="Generating signals"):
        news_items = group.to_dict('records')
        
        if simulation_mode:
            # Use simulation
            prediction = simulate_model_prediction(news_items)
        else:
            # Use actual model
            prompt = create_prompt_from_news(ticker, news_items)
            
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                max_length=2048,
                truncation=True,
                padding=True
            ).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.MAX_NEW_TOKENS,
                    temperature=config.TEMPERATURE,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse response to extract prediction
            prediction = parse_model_response(response)
        
        # Create signal
        signal = {
            'ticker': ticker,
            'gvkey': gvkey,
            'period': str(period),
            'date': period.to_timestamp().strftime('%Y-%m-%d'),
            'direction': prediction['direction'],
            'estimated_change': prediction['estimated_change'],
            'confidence': prediction.get('confidence', 0.5),
            'num_news': len(news_items),
            'signal': 1 if prediction['direction'] == 'up' else (-1 if prediction['direction'] == 'down' else 0)
        }
        
        signals.append(signal)
    
    # Convert to DataFrame
    signals_df = pd.DataFrame(signals)
    
    print(f"\nGenerated {len(signals_df)} trading signals")
    print(f"Signal distribution:")
    print(signals_df['signal'].value_counts())
    
    # Save signals
    signals_file = config.RESULTS_DIR / "trading_signals.csv"
    signals_df.to_csv(signals_file, index=False)
    print(f"\nSaved trading signals to: {signals_file}")
    
    # Display summary
    print(f"Total signals: {len(signals_df)}")
    print(f"Bullish signals: {(signals_df['signal'] == 1).sum()}")
    print(f"Bearish signals: {(signals_df['signal'] == -1).sum()}")
    print(f"Neutral signals: {(signals_df['signal'] == 0).sum()}")
    print(f"\nAverage estimated return: {signals_df['estimated_change'].mean():.2f}%")
    print(f"Top 5 bullish predictions:")
    print(signals_df.nlargest(5, 'estimated_change')[['ticker', 'date', 'estimated_change']])
    
    return True

def parse_model_response(response):
    # Simple parsing - look for direction and percentage
    response_lower = response.lower()
    
    if 'direction: up' in response_lower or 'direction:up' in response_lower:
        direction = 'up'
    elif 'direction: down' in response_lower or 'direction:down' in response_lower:
        direction = 'down'
    else:
        direction = 'neutral'
    
    # Try to extract percentage
    import re
    percentage_match = re.search(r'(\d+\.?\d*)\s*%', response)
    if percentage_match:
        estimated_change = float(percentage_match.group(1))
        if direction == 'down':
            estimated_change = -estimated_change
    else:
        estimated_change = 2.0 if direction == 'up' else (-2.0 if direction == 'down' else 0.0)
    
    return {
        'direction': direction,
        'estimated_change': estimated_change,
        'confidence': 0.7
    }

def main():
    success = generate_signals()
    
    if success:
        print("\n3 completed successfully!")
        print(f"Trading signals saved to: {config.RESULTS_DIR / 'trading_signals.csv'}")
        return True
    else:
        print("\nStep 3 failed!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
