import pandas as pd
import requests
import json
from pathlib import Path

OLLAMA_URL = "http://localhost:11434"
DATA_PATH = "data"
DASHBOARD_PATH = "data/dashboards"

print("Starting Customer Intelligence Analysis...")

def load_and_sample_data():
    """Load data and create manageable summaries for LLM"""

    original_data = pd.read_csv(f"{DATA_PATH}/customer_intelligence_dataset.csv")
    print(f"Original dataset: {len(original_data)} rows")

    churn_data = pd.read_csv(f"{DASHBOARD_PATH}/churn_dashboard.csv")
    segments_data = pd.read_csv(f"{DASHBOARD_PATH}/customer_segment_dashboard.csv")
    sales_forecast = pd.read_csv(f"{DASHBOARD_PATH}/sales_forecast_future_2024_2025.csv")

    print(f"Loaded all data files")
    return original_data, churn_data, segments_data, sales_forecast


def create_sales_summary(df):
    """Create sales summary from original dataset"""
    
    df['sale_date'] = pd.to_datetime(df['sale_date'])
    df['total_value'] = df['price'] * df['quantity']

    monthly_sales = df.groupby(df['sale_date'].dt.to_period('M'))['total_value'].sum()

    category_sales = df.groupby('category')['total_value'].sum().sort_values(ascending=False)

    top_products = df.groupby('product_name')['total_value'].sum().sort_values(ascending=False).head(10)

    avg_monthly = monthly_sales.mean()
    spikes = monthly_sales[monthly_sales > avg_monthly * 1.2]
    
    summary = {
        "total_sales": float(df['total_value'].sum()),
        "total_transactions": len(df),
        "avg_transaction_value": float(df['total_value'].mean()),
        "date_range": f"{df['sale_date'].min().date()} to {df['sale_date'].max().date()}",
        "top_3_categories": category_sales.head(3).to_dict(),
        "monthly_sales_trend": monthly_sales.tail(12).to_dict(),  # Last 12 months
        "top_5_products": top_products.head(5).to_dict(),
        "peak_sales_month": monthly_sales.idxmax().strftime('%Y-%m'),
        "peak_sales_value": float(monthly_sales.max()),
        "sales_spikes": {str(k): float(v) for k, v in spikes.to_dict().items()},
        "avg_monthly_sales": float(avg_monthly),
        "growth_trend": "increasing" if monthly_sales.iloc[-1] > monthly_sales.iloc[0] else "decreasing"
    }
    
    return summary

def create_segments_summary(segments_df):
    """Summarize customer segments"""
    segment_counts = segments_df['segment_name'].value_counts()
    
    return {
        "total_customers": len(segments_df),
        "segment_distribution": segment_counts.to_dict(),
        "largest_segment": segment_counts.index[0],
        "smallest_segment": segment_counts.index[-1]
    }

def create_churn_summary(churn_df):
    """Summarize churn predictions"""
    churn_rate = (churn_df['churn'] == 'Yes').mean() * 100
    churn_count = (churn_df['churn'] == 'Yes').sum()
    
    return {
        "total_customers": len(churn_df),
        "churn_rate_percent": round(churn_rate, 1),
        "customers_will_churn": int(churn_count),
        "customers_retained": int(len(churn_df) - churn_count),
        "churn_risk_level": "High" if churn_rate > 25 else "Medium" if churn_rate > 15 else "Low"
    }

def create_forecast_summary(forecast_df):
    """Summarize sales forecasting"""
    return {
        "forecast_months": len(forecast_df),
        "total_predicted_sales": float(forecast_df['predicted_sales'].sum()),
        "avg_monthly_forecast": float(forecast_df['predicted_sales'].mean()),
        "highest_month": forecast_df.loc[forecast_df['predicted_sales'].idxmax(), 'date'],
        "highest_value": float(forecast_df['predicted_sales'].max())
    }


def query_llm(prompt):
    """Improved function to query Ollama with better error handling"""
    try:
        print("⏳ Sending request to Mistral (this may take 1-3 minutes)...")
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": "mistral:7b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 1500 
                }
            },
            timeout=300 
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"Error: HTTP {response.status_code} - {response.text}"
            
    except requests.exceptions.Timeout:
        return "Error: Request timed out. Mistral may be processing a complex prompt."
    except Exception as e:
        return f"Error connecting to Ollama: {str(e)}"


def create_simple_insights_prompt(sales_summary, segments_summary, churn_summary, forecast_summary):
    """Create a simpler, more focused prompt for faster processing"""
    
    prompt = f"""
You are a business analyst. Analyze this customer intelligence data and provide clear insights.

## KEY BUSINESS METRICS
Sales Performance:
- Total Sales: ${sales_summary['total_sales']:,.0f}
- Transactions: {sales_summary['total_transactions']:,}
- Average Order: ${sales_summary['avg_transaction_value']:.0f}
- Peak Month: {sales_summary['peak_sales_month']} (${sales_summary['peak_sales_value']:,.0f})
- Trend: {sales_summary['growth_trend']}

Top Categories: {list(sales_summary['top_3_categories'].keys())[:3]}

Customer Segments:
- Total Customers: {segments_summary['total_customers']:,}
- Largest Segment: {segments_summary['largest_segment']} 
- Segments: {list(segments_summary['segment_distribution'].keys())}

Churn Analysis:
- Churn Rate: {churn_summary['churn_rate_percent']}% ({churn_summary['churn_risk_level']} Risk)
- At-Risk Customers: {churn_summary['customers_will_churn']:,}

Sales Forecast:
- Next {forecast_summary['forecast_months']} months: ${forecast_summary['total_predicted_sales']:,.0f}
- Monthly Average: ${forecast_summary['avg_monthly_forecast']:,.0f}

## ANALYSIS REQUIRED:
1. Sales Performance: What are the key trends and opportunities?
2. Customer Segments: What do these segments mean and why were they created?
3. Churn Risk: How serious is {churn_summary['churn_rate_percent']}% churn? What should be done?
4. Sales Forecast: Is this realistic? What does it suggest?
5. Top 3 Recommendations: What should the business focus on?

Keep analysis practical and business-focused. Use bullet points for clarity.
"""
    return prompt


original_data, churn_data, segments_data, sales_forecast = load_and_sample_data()

print("Creating sales summary...")
sales_summary = create_sales_summary(original_data)

print("Creating segments summary...")
segments_summary = create_segments_summary(segments_data)

print("Creating churn summary...")
churn_summary = create_churn_summary(churn_data)

print("Creating forecast summary...")
forecast_summary = create_forecast_summary(sales_forecast)

print("Creating simplified LLM prompt...")
simple_prompt = create_simple_insights_prompt(sales_summary, segments_summary, churn_summary, forecast_summary)

print("Querying Mistral for insights (simple analysis)...")
insights = query_llm(simple_prompt)

if "Error" not in insights and len(insights) > 100:
    print("Simple analysis successful!")
    


print("\n" + "="*80)
print("CUSTOMER INTELLIGENCE INSIGHTS (Mistral 7B)")
print("="*80)
print(insights)

with open("customer_insights_mistral.txt", "w") as f:
    f.write("CUSTOMER INTELLIGENCE INSIGHTS (Mistral 7B)\n")
    f.write("="*50 + "\n\n")
    f.write(insights)

print(f"\nInsights saved to: customer_insights_mistral.txt")

print(f"\nDEBUG INFO:")
print(f"Total Sales: ${sales_summary['total_sales']:,.0f}")
print(f"Churn Rate: {churn_summary['churn_rate_percent']}%")
print(f"Largest Segment: {segments_summary['largest_segment']}")
print(f"Sales Trend: {sales_summary['growth_trend']}")
print(f"Forecast: ${forecast_summary['total_predicted_sales']:,.0f} over {forecast_summary['forecast_months']} months")

print("\nAnalysis complete!")