import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import json
import numpy as np
import google.generativeai as genai
from plotly.subplots import make_subplots
import time
from io import StringIO
try:
    import ofxparse
    OFX_AVAILABLE = True
except ImportError:
    OFX_AVAILABLE = False
    print("Warning: ofxparse not installed. QFX import will not work. Install with: pip install ofxparse")


# Add this at the top with other session state initializations
if 'last_ai_call' not in st.session_state:
    st.session_state.last_ai_call = 0


def ai_chat(user_message, context):
    """AI-powered financial chat assistant with rate limiting"""
    if not AI_ENABLED:
        return "Please set up GEMINI_API_KEY to use AI chat features."

    # Rate limiting: Wait at least 2 seconds between calls
    time_since_last_call = time.time() - st.session_state.last_ai_call
    if time_since_last_call < 2:
        time.sleep(2 - time_since_last_call)

    prompt = f"""You are Origin Financial's AI assistant, helping users with their personal finances.

USER'S FINANCIAL CONTEXT:
{context}

CHAT HISTORY:
{st.session_state.chat_history[-5:] if st.session_state.chat_history else 'No previous messages'}

USER MESSAGE: {user_message}

Provide a helpful, concise response. Use specific numbers from their data when relevant. Be encouraging and actionable."""

    try:
        response = model.generate_content(prompt)
        st.session_state.last_ai_call = time.time()
        return response.text.strip()
    except Exception as e:
        if "429" in str(e):
            return "⚠️ Rate limit reached. Please wait a moment and try again."
        return f"I'm having trouble processing that. Error: {str(e)}"


def ai_financial_insights(data_summary):
    """Generate insights with rate limiting"""
    if not AI_ENABLED:
        return ["Connect AI (set GEMINI_API_KEY) for personalized insights",
                "Track spending to get recommendations",
                "Set budgets to monitor your finances"]

    # Rate limiting
    time_since_last_call = time.time() - st.session_state.last_ai_call
    if time_since_last_call < 2:
        time.sleep(2 - time_since_last_call)

    prompt = f"""You are a professional financial advisor analyzing a user's financial data. Provide 5-7 specific, actionable insights and recommendations.

FINANCIAL DATA:
{data_summary}

Provide insights on:
1. Spending patterns and anomalies
2. Savings opportunities
3. Budget optimization
4. Debt management (if applicable)
5. Investment suggestions
6. Risk assessment

Format each insight as a bullet point starting with an emoji. Be specific with numbers and percentages. Keep insights concise and actionable."""

    try:
        response = model.generate_content(prompt)
        st.session_state.last_ai_call = time.time()
        insights = response.text.strip().split('\n')
        return [i.strip() for i in insights if i.strip()]
    except Exception as e:
        if "429" in str(e):
            return ["⚠️ Rate limit reached. Please wait a moment before requesting insights again."]
        return [f"Unable to generate insights: {str(e)}"]


# Page config
st.set_page_config(
    page_title="Origin - AI Financial Planning",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)

# Gemini API setup
GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY", "")
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel('gemini-2.0-flash-lite')
    AI_ENABLED = True
except:
    AI_ENABLED = False

# File paths
TRANSACTIONS_FILE = "transactions.csv"
LOANS_FILE = "loans.csv"
ACCOUNTS_FILE = "accounts.csv"
BUDGETS_FILE = "budgets.csv"
GOALS_FILE = "goals.csv"
RECURRING_FILE = "recurring.csv"

# Initialize session state
if 'refresh_trigger' not in st.session_state:
    st.session_state.refresh_trigger = 0
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'


def trigger_refresh():
    st.session_state.refresh_trigger += 1

# Data management functions


def load_transactions():
    if os.path.exists(TRANSACTIONS_FILE):
        df = pd.read_csv(TRANSACTIONS_FILE)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Ensure linking columns exist
        if 'Linked_Recurring_ID' not in df.columns:
            df['Linked_Recurring_ID'] = ''
        if 'Linked_Loan_ID' not in df.columns:
            df['Linked_Loan_ID'] = ''
        
        return df
    return pd.DataFrame(columns=[
        'Date', 'Description', 'Amount', 'Category', 'Type', 
        'Account', 'Source', 'Tags', 'Linked_Recurring_ID', 'Linked_Loan_ID'
    ])

def load_loans():
    if os.path.exists(LOANS_FILE):
        df = pd.read_csv(LOANS_FILE)
        df['Start_Date'] = pd.to_datetime(df['Start_Date'])
        return df
    return pd.DataFrame(columns=['Description', 'Principal', 'Rate', 'Term_Years', 'Start_Date', 'Monthly_Payment', 'Remaining', 'Type'])


def load_accounts():
    if os.path.exists(ACCOUNTS_FILE):
        return pd.read_csv(ACCOUNTS_FILE)
    return pd.DataFrame(columns=['Account_Name', 'Account_Type', 'Balance', 'Institution', 'Currency', 'Interest_Rate'])

# def load_budgets():
#     if os.path.exists(BUDGETS_FILE):
#         return pd.read_csv(BUDGETS_FILE)
#     default_budgets = pd.DataFrame({
#         'Category': ['Food & Dining', 'Shopping', 'Transportation', 'Entertainment', 'Bills & Utilities', 'Healthcare', 'Education', 'Personal Care', 'Other'],
#         'Budget': [800, 500, 300, 200, 400, 200, 300, 150, 300],
#         'Alert_Threshold': [90, 90, 90, 90, 90, 90, 90, 90, 90]
#     })
#     return default_budgets


def load_budgets():
    if os.path.exists(BUDGETS_FILE):
        df = pd.read_csv(BUDGETS_FILE)
        # Ensure Alert_Threshold column exists
        if 'Alert_Threshold' not in df.columns:
            df['Alert_Threshold'] = 90
        return df

    # Default budgets with Alert_Threshold
    default_budgets = pd.DataFrame({
        'Category': ['Food & Dining', 'Shopping', 'Transportation', 'Entertainment',
                     'Bills & Utilities', 'Healthcare', 'Education', 'Personal Care', 'Other'],
        'Budget': [800, 500, 300, 200, 400, 200, 300, 150, 300],
        'Alert_Threshold': [90, 90, 90, 90, 90, 90, 90, 90, 90]
    })
    return default_budgets


def load_goals():
    if os.path.exists(GOALS_FILE):
        df = pd.read_csv(GOALS_FILE)
        df['Target_Date'] = pd.to_datetime(df['Target_Date'])
        return df
    return pd.DataFrame(columns=['Goal_Name', 'Target_Amount', 'Current_Amount', 'Target_Date', 'Priority', 'Category'])


def load_recurring():
    if os.path.exists(RECURRING_FILE):
        df = pd.read_csv(RECURRING_FILE)
        df['Next_Date'] = pd.to_datetime(df['Next_Date'])
        return df
    return pd.DataFrame(columns=['Description', 'Amount', 'Category', 'Frequency', 'Next_Date', 'Account', 'Active'])


# Add these imports at the top

def parse_qfx_file(uploaded_file):
    """Parse QFX/OFX file and return DataFrame"""
    try:
        # Read the file content
        file_content = uploaded_file.read()
        
        # Try to decode
        try:
            content_str = file_content.decode('utf-8')
        except:
            content_str = file_content.decode('latin-1')
        
        # Parse OFX/QFX
        ofx = ofxparse.OfxParser.parse(StringIO(content_str))
        
        transactions = []
        account_id = None
        
        # Extract account information
        if hasattr(ofx, 'account'):
            account_id = ofx.account.account_id if hasattr(ofx.account, 'account_id') else None
        elif hasattr(ofx, 'accounts') and ofx.accounts:
            account_id = ofx.accounts[0].account_id if hasattr(ofx.accounts[0], 'account_id') else None
        
        # Extract transactions from account
        if hasattr(ofx, 'account') and hasattr(ofx.account, 'statement'):
            for trans in ofx.account.statement.transactions:
                transactions.append({
                    'Date': trans.date,
                    'Description': trans.memo or trans.payee or 'Unknown',
                    'Amount': float(trans.amount),
                    'Type': 'income' if trans.amount > 0 else 'expense',
                    'Transaction_ID': trans.id if hasattr(trans, 'id') else ''
                })
        elif hasattr(ofx, 'accounts'):
            for account in ofx.accounts:
                if hasattr(account, 'statement'):
                    for trans in account.statement.transactions:
                        transactions.append({
                            'Date': trans.date,
                            'Description': trans.memo or trans.payee or 'Unknown',
                            'Amount': float(trans.amount),
                            'Type': 'income' if trans.amount > 0 else 'expense',
                            'Transaction_ID': trans.id if hasattr(trans, 'id') else ''
                        })
        
        df = pd.DataFrame(transactions)
        
        return df, account_id
    
    except Exception as e:
        raise Exception(f"Error parsing QFX file: {str(e)}")

def parse_csv_file(uploaded_file):
    """Parse CSV file and return DataFrame"""
    try:
        df = pd.read_csv(uploaded_file)
        return df, None
    except Exception as e:
        raise Exception(f"Error parsing CSV file: {str(e)}")

def save_transactions(df):
    df.to_csv(TRANSACTIONS_FILE, index=False)
    trigger_refresh()


def save_loans(df):
    df.to_csv(LOANS_FILE, index=False)
    trigger_refresh()


def save_accounts(df):
    df.to_csv(ACCOUNTS_FILE, index=False)
    trigger_refresh()


def save_budgets(df):
    df.to_csv(BUDGETS_FILE, index=False)
    trigger_refresh()


def save_goals(df):
    df.to_csv(GOALS_FILE, index=False)
    trigger_refresh()


def save_recurring(df):
    df.to_csv(RECURRING_FILE, index=False)
    trigger_refresh()


def update_transaction(idx, updates):
    """Update a specific transaction"""
    global transactions_df
    for key, value in updates.items():
        transactions_df.at[idx, key] = value
    save_transactions(transactions_df)

# Financial calculations


def calculate_net_worth(transactions_df, accounts_df, loans_df):
    assets = accounts_df['Balance'].sum() if not accounts_df.empty else 0
    liabilities = loans_df['Remaining'].sum(
    ) if not loans_df.empty and 'Remaining' in loans_df.columns else 0
    return assets - liabilities


def calculate_monthly_loan_payment(principal, rate, term_years):
    if rate == 0 or term_years == 0:
        return principal / (term_years * 12) if term_years > 0 else 0
    monthly_rate = rate / 12 / 100
    n = term_years * 12
    payment = principal * (monthly_rate * (1 + monthly_rate)
                           ** n) / ((1 + monthly_rate)**n - 1)
    return payment


def calculate_compound_interest(principal, rate, years, monthly_contribution=0):
    """Calculate future value with compound interest"""
    months = years * 12
    monthly_rate = rate / 12 / 100

    # Future value of principal
    fv_principal = principal * (1 + monthly_rate) ** months

    # Future value of monthly contributions
    if monthly_contribution > 0:
        fv_contributions = monthly_contribution * \
            (((1 + monthly_rate) ** months - 1) / monthly_rate)
    else:
        fv_contributions = 0

    return fv_principal + fv_contributions

# def get_spending_insights(transactions_df, budgets_df):
#     current_month = datetime.now().replace(day=1)
#     monthly_transactions = transactions_df[
#         (transactions_df['Date'] >= current_month) &
#         (transactions_df['Type'] == 'expense')
#     ]

#     insights = []
#     if not monthly_transactions.empty:
#         category_spending = monthly_transactions.groupby('Category')['Amount'].sum().abs()

#         for category in category_spending.index:
#             spent = category_spending[category]
#             budget_row = budgets_df[budgets_df['Category'] == category]
#             if not budget_row.empty:
#                 budget = budget_row['Budget'].values[0]
#                 threshold = budget_row.get('Alert_Threshold', [90])[0]
#                 if spent > budget * (threshold / 100):
#                     insights.append({
#                         'type': 'warning',
#                         'category': category,
#                         'message': f"You're at {spent/budget*100:.0f}% of your {category} budget",
#                         'spent': spent,
#                         'budget': budget
#                     })

#     return insights


def get_spending_insights(transactions_df, budgets_df):
    """Get insights about spending relative to budget"""
    current_month = datetime.now().replace(day=1)
    monthly_transactions = transactions_df[
        (transactions_df['Date'] >= current_month) &
        (transactions_df['Type'] == 'expense')
    ]

    insights = []
    if not monthly_transactions.empty and not budgets_df.empty:
        category_spending = monthly_transactions.groupby('Category')[
            'Amount'].sum().abs()

        for category in category_spending.index:
            spent = category_spending[category]
            budget_row = budgets_df[budgets_df['Category'] == category]

            if not budget_row.empty:
                budget = budget_row['Budget'].values[0]

                # Safely access Alert_Threshold
                if 'Alert_Threshold' in budget_row.columns:
                    threshold = budget_row['Alert_Threshold'].values[0]
                else:
                    threshold = 90  # Default threshold if column doesn't exist

                # Check if spending exceeds threshold
                if spent > budget * (threshold / 100):
                    insights.append({
                        'type': 'warning',
                        'category': category,
                        'message': f"You're at {spent/budget*100:.0f}% of your {category} budget",
                        'spent': spent,
                        'budget': budget
                    })

    return insights


def predict_end_of_month_spending(transactions_df):
    """Predict spending by end of month based on current trends"""
    current_month_start = datetime.now().replace(day=1)
    current_day = datetime.now().day
    days_in_month = (current_month_start.replace(
        month=current_month_start.month % 12 + 1, day=1) - timedelta(days=1)).day

    monthly_transactions = transactions_df[
        (transactions_df['Date'] >= current_month_start) &
        (transactions_df['Type'] == 'expense')
    ]

    if monthly_transactions.empty:
        return 0

    current_spending = abs(monthly_transactions['Amount'].sum())
    daily_average = current_spending / current_day
    predicted_total = daily_average * days_in_month

    return predicted_total

# AI Functions with Gemini


def ai_categorize_transaction(description, amount):
    if not AI_ENABLED:
        return "Other"

    categories = ['Food & Dining', 'Shopping', 'Transportation', 'Entertainment',
                  'Bills & Utilities', 'Healthcare', 'Education', 'Personal Care', 'Income', 'Other']

    prompt = f"""Categorize this financial transaction into ONE of these categories: {', '.join(categories)}

Transaction: "{description}"
Amount: ${amount}

Respond with ONLY the category name, nothing else."""

    try:
        response = model.generate_content(prompt)
        result = response.text.strip()

        for cat in categories:
            if cat.lower() in result.lower():
                return cat
        return "Other"
    except Exception as e:
        print(f"AI Error: {e}")
        return "Other"


def ai_categorize_transactions_batch(transactions_list):
    """Categorize multiple transactions in a single AI call to avoid rate limits"""
    if not AI_ENABLED or not transactions_list:
        return ["Other"] * len(transactions_list)

    categories = ['Food & Dining', 'Shopping', 'Transportation', 'Entertainment',
                  'Bills & Utilities', 'Healthcare', 'Education', 'Personal Care', 'Income', 'Other']

    # Build a single prompt for all transactions
    transactions_text = "\n".join([
        f"{i+1}. Description: '{row['Description']}', Amount: ${row['Amount']}"
        for i, row in enumerate(transactions_list)
    ])

    prompt = f"""Categorize these {len(transactions_list)} financial transactions into ONE of these categories: {', '.join(categories)}

Transactions:
{transactions_text}

Respond with ONLY a comma-separated list of category names in the same order as the transactions, nothing else.
Example format: Food & Dining,Transportation,Shopping,Income"""

    try:
        response = model.generate_content(prompt)
        result = response.text.strip()

        # Parse the comma-separated response
        categorized = [cat.strip() for cat in result.split(',')]

        # Validate and clean up categories
        final_categories = []
        for cat in categorized:
            matched = False
            for valid_cat in categories:
                if valid_cat.lower() in cat.lower():
                    final_categories.append(valid_cat)
                    matched = True
                    break
            if not matched:
                final_categories.append("Other")

        # Ensure we have the right number of categories
        while len(final_categories) < len(transactions_list):
            final_categories.append("Other")

        return final_categories[:len(transactions_list)]

    except Exception as e:
        print(f"AI Batch Categorization Error: {e}")
        # Fallback to simple categorization
        return [ai_categorize_transaction_simple(row['Description'], row['Amount'])
                for row in transactions_list]


def ai_financial_insights(data_summary):
    if not AI_ENABLED:
        return ["Connect AI (set GEMINI_API_KEY) for personalized insights", "Track spending to get recommendations", "Set budgets to monitor your finances"]

    prompt = f"""You are a professional financial advisor analyzing a user's financial data. Provide 5-7 specific, actionable insights and recommendations.

FINANCIAL DATA:
{data_summary}

Provide insights on:
1. Spending patterns and anomalies
2. Savings opportunities
3. Budget optimization
4. Debt management (if applicable)
5. Investment suggestions
6. Risk assessment

Format each insight as a bullet point starting with an emoji. Be specific with numbers and percentages. Keep insights concise and actionable."""

    try:
        response = model.generate_content(prompt)
        insights = response.text.strip().split('\n')
        return [i.strip() for i in insights if i.strip()]
    except Exception as e:
        return [f"Unable to generate insights: {str(e)}"]


def ai_categorize_transaction_simple(description, amount):
    """Simple rule-based categorization fallback"""
    description_lower = description.lower()

    # Simple keyword matching
    if any(word in description_lower for word in ['grocery', 'restaurant', 'food', 'cafe', 'starbucks', 'mcdonald']):
        return 'Food & Dining'
    elif any(word in description_lower for word in ['gas', 'uber', 'lyft', 'transit', 'parking', 'taxi']):
        return 'Transportation'
    elif any(word in description_lower for word in ['amazon', 'store', 'shop', 'mall', 'retail']):
        return 'Shopping'
    elif any(word in description_lower for word in ['netflix', 'spotify', 'movie', 'theater', 'game']):
        return 'Entertainment'
    elif any(word in description_lower for word in ['electric', 'water', 'gas bill', 'internet', 'phone bill', 'utility']):
        return 'Bills & Utilities'
    elif any(word in description_lower for word in ['doctor', 'hospital', 'pharmacy', 'medical', 'health']):
        return 'Healthcare'
    elif any(word in description_lower for word in ['school', 'tuition', 'course', 'book', 'education']):
        return 'Education'
    elif any(word in description_lower for word in ['salon', 'gym', 'spa', 'personal']):
        return 'Personal Care'
    elif amount > 0:
        return 'Income'
    else:
        return 'Other'


def ai_chat(user_message, context):
    """AI-powered financial chat assistant"""
    if not AI_ENABLED:
        return "Please set up GEMINI_API_KEY to use AI chat features."

    prompt = f"""You are Origin Financial's AI assistant, helping users with their personal finances.

USER'S FINANCIAL CONTEXT:
{context}

CHAT HISTORY:
{st.session_state.chat_history[-5:] if st.session_state.chat_history else 'No previous messages'}

USER MESSAGE: {user_message}

Provide a helpful, concise response. Use specific numbers from their data when relevant. Be encouraging and actionable."""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"I'm having trouble processing that. Error: {str(e)}"


def ai_suggest_budget(transactions_df, income):
    """AI-powered budget suggestions based on spending history"""
    if not AI_ENABLED or transactions_df.empty:
        return None

    category_spending = transactions_df[transactions_df['Type'] == 'expense'].groupby(
        'Category')['Amount'].sum().abs()

    prompt = f"""As a financial advisor, suggest an optimal monthly budget allocation based on this spending history and income.

Monthly Income: ${income:,.2f}

Historical Spending by Category:
{category_spending.to_dict()}

Provide a recommended budget for each category using the 50/30/20 rule (50% needs, 30% wants, 20% savings).
Format as: Category: $amount (brief reason)"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except:
        return None


# Enhanced Custom CSS - True Origin Financial Style
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    :root {
        --primary: #6366f1;
        --primary-dark: #4f46e5;
        --success: #10b981;
        --warning: #f59e0b;
        --danger: #ef4444;
        --bg-primary: #0a0a0a;
        --bg-secondary: #141414;
        --bg-tertiary: #1a1a1a;
        --text-primary: #ffffff;
        --text-secondary: #a3a3a3;
        --border: #262626;
    }
    
    .main {
        background: var(--bg-primary);
        color: var(--text-primary);
    }
    
    .stApp {
        background: linear-gradient(180deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border);
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        display: none;
    }
    
    [data-testid="stSidebar"] [data-baseweb="radio"] label {
        background: transparent;
        padding: 12px 16px;
        border-radius: 8px;
        transition: all 0.2s ease;
        color: var(--text-secondary);
        font-weight: 500;
        margin: 4px 0;
    }
    
    [data-testid="stSidebar"] [data-baseweb="radio"] label:hover {
        background: var(--bg-tertiary);
        color: var(--text-primary);
    }
    
    [data-testid="stSidebar"] [data-baseweb="radio"] [data-checked="true"] label {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        font-weight: 600;
    }
    
    /* Glass morphism cards */
    .glass-card {
        background: rgba(26, 26, 26, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, var(--bg-tertiary) 0%, var(--bg-secondary) 100%);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid var(--border);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(99, 102, 241, 0.2);
    }
    
    .insight-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(79, 70, 229, 0.05) 100%);
        border-left: 4px solid var(--primary);
        padding: 16px;
        border-radius: 12px;
        margin: 12px 0;
        transition: all 0.2s ease;
    }
    
    .insight-card:hover {
        transform: translateX(4px);
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(79, 70, 229, 0.08) 100%);
    }
    
    .warning-card {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(220, 38, 38, 0.05) 100%);
        border-left: 4px solid var(--danger);
        padding: 16px;
        border-radius: 12px;
        margin: 12px 0;
    }
    
    .success-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%);
        border-left: 4px solid var(--success);
        padding: 16px;
        border-radius: 12px;
        margin: 12px 0;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 36px;
        font-weight: 800;
        color: var(--text-primary);
        letter-spacing: -0.5px;
    }
    
    [data-testid="stMetricLabel"] {
        color: var(--text-secondary);
        font-size: 14px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 14px;
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        cursor: pointer;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
    }
    
    .stButton > button:active {
        transform: translateY(0);
    }
    
    /* Secondary button style */
    .stButton > button[kind="secondary"] {
        background: var(--bg-tertiary);
        border: 1px solid var(--border);
        box-shadow: none;
    }
    
    /* Inputs */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select,
    .stDateInput > div > div > input,
    .stTextArea textarea {
        background: var(--bg-tertiary);
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-radius: 10px;
        font-size: 14px;
        padding: 12px;
        transition: all 0.2s ease;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stTextArea textarea:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, var(--primary) 0%, var(--primary-dark) 100%);
        border-radius: 10px;
        height: 8px;
    }
    
    .stProgress > div > div {
        background: var(--bg-tertiary);
        border-radius: 10px;
        height: 8px;
    }
    
    /* Headers */
    h1 {
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 8px;
        letter-spacing: -1px;
        background: linear-gradient(135deg, var(--text-primary) 0%, var(--text-secondary) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    h2 {
        font-size: 28px;
        font-weight: 700;
        margin-bottom: 16px;
        letter-spacing: -0.5px;
    }
    
    h3 {
        font-size: 20px;
        font-weight: 600;
        margin-bottom: 12px;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: var(--bg-tertiary);
        border-radius: 12px;
        padding: 6px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-secondary);
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text-primary);
        background: rgba(255, 255, 255, 0.05);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
    }
    
    /* Dataframe */
    .stDataFrame {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: var(--bg-tertiary);
        border-radius: 10px;
        color: var(--text-primary);
        font-weight: 600;
        padding: 16px;
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--bg-secondary);
    }
    
    /* File uploader */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed var(--border);
        border-radius: 12px;
        background: var(--bg-tertiary);
        padding: 32px;
        transition: all 0.2s ease;
    }
    
    [data-testid="stFileUploadDropzone"]:hover {
        border-color: var(--primary);
        background: rgba(99, 102, 241, 0.05);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--primary);
    }
    
    /* Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .metric-card, .insight-card, .warning-card, .success-card {
        animation: fadeIn 0.3s ease;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    # header {visibility: hidden;}
            
    
    /* Chat messages */
    .chat-message {
        padding: 16px;
        border-radius: 12px;
        margin: 12px 0;
        animation: fadeIn 0.3s ease;
    }
    
    .user-message {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        margin-left: 20%;
    }
    
    .ai-message {
        background: var(--bg-tertiary);
        margin-right: 20%;
        border: 1px solid var(--border);
    }
</style>
""", unsafe_allow_html=True)

# Load all data
transactions_df = load_transactions()
loans_df = load_loans()
accounts_df = load_accounts()
budgets_df = load_budgets()
goals_df = load_goals()
recurring_df = load_recurring()

# Sidebar Navigation
with st.sidebar:
    st.markdown("### 🏦 Origin Financial")
    st.markdown("*AI-Powered Finance*")
    st.markdown("---")

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "💰 Accounts", "📊 Transactions", "💳 Budgets",
         "🎯 Goals", "🏦 Loans", "🔄 Recurring", "🤖 AI Assistant", "📈 Analytics"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Quick stats in sidebar
    net_worth = calculate_net_worth(transactions_df, accounts_df, loans_df)
    st.metric("Net Worth", f"${net_worth:,.0f}")

    if not transactions_df.empty:
        current_month_start = datetime.now().replace(day=1)
        current_month_transactions = transactions_df[
            (transactions_df['Date'] >= current_month_start) &
            (transactions_df['Type'] == 'expense')
        ]
        monthly_spending = abs(current_month_transactions['Amount'].sum())
        st.metric("This Month", f"${monthly_spending:,.0f}")

        # Predicted spending
        predicted = predict_end_of_month_spending(transactions_df)
        if predicted > 0:
            st.metric("Predicted EOD", f"${predicted:,.0f}")

    # AI Status
    st.markdown("---")
    ai_status = "🟢 Connected" if AI_ENABLED else "🔴 Disconnected"
    st.markdown(f"**AI Status:** {ai_status}")

# DASHBOARD PAGE
if page == "🏠 Dashboard":
    st.title("Financial Dashboard")
    st.markdown("*Your complete financial overview*")

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)

    net_worth = calculate_net_worth(transactions_df, accounts_df, loans_df)

    # Calculate monthly stats
    current_month_start = datetime.now().replace(day=1)
    last_month_start = (current_month_start - timedelta(days=1)).replace(day=1)

    current_income = transactions_df[
        (transactions_df['Date'] >= current_month_start) &
        (transactions_df['Type'] == 'income')
    ]['Amount'].sum()

    current_expenses = abs(transactions_df[
        (transactions_df['Date'] >= current_month_start) &
        (transactions_df['Type'] == 'expense')
    ]['Amount'].sum())

    last_month_expenses = abs(transactions_df[
        (transactions_df['Date'] >= last_month_start) &
        (transactions_df['Date'] < current_month_start) &
        (transactions_df['Type'] == 'expense')
    ]['Amount'].sum())

    expense_change = ((current_expenses - last_month_expenses) /
                      last_month_expenses * 100) if last_month_expenses > 0 else 0

    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Net Worth", f"${net_worth:,.0f}",
                  delta=f"${abs(net_worth):,.0f}" if net_worth > 0 else None)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Monthly Income", f"${current_income:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Monthly Spending", f"${current_expenses:,.0f}",
                  delta=f"{expense_change:+.1f}%",
                  delta_color="inverse")
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        savings = current_income - current_expenses
        savings_rate = (savings / current_income *
                        100) if current_income > 0 else 0
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Savings Rate",
                  f"{savings_rate:.1f}%", delta=f"${savings:,.0f}")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Main content area
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Cash Flow Chart
        st.subheader("💹 Cash Flow Analysis")

        if not transactions_df.empty:
            # Last 6 months
            six_months_ago = datetime.now() - timedelta(days=180)
            recent_data = transactions_df[transactions_df['Date'] >= six_months_ago].copy(
            )
            recent_data['Month'] = recent_data['Date'].dt.to_period(
                'M').astype(str)

            monthly_summary = recent_data.groupby(['Month', 'Type'])[
                'Amount'].sum().reset_index()
            monthly_summary['Amount'] = monthly_summary.apply(
                lambda x: abs(
                    x['Amount']) if x['Type'] == 'expense' else x['Amount'],
                axis=1
            )

            # Create subplots
            fig = make_subplots(
                rows=1, cols=1,
                specs=[[{"secondary_y": True}]]
            )

            income_data = monthly_summary[monthly_summary['Type'] == 'income']
            expense_data = monthly_summary[monthly_summary['Type'] == 'expense']

            fig.add_trace(
                go.Bar(name='Income', x=income_data['Month'], y=income_data['Amount'],
                       marker_color='#10b981'),
                secondary_y=False
            )

            fig.add_trace(
                go.Bar(name='Expenses', x=expense_data['Month'], y=expense_data['Amount'],
                       marker_color='#ef4444'),
                secondary_y=False
            )

            # Calculate net and add as line
            months = monthly_summary['Month'].unique()
            net_by_month = []
            for month in months:
                income = monthly_summary[(monthly_summary['Month'] == month) & (
                    monthly_summary['Type'] == 'income')]['Amount'].sum()
                expense = monthly_summary[(monthly_summary['Month'] == month) & (
                    monthly_summary['Type'] == 'expense')]['Amount'].sum()
                net_by_month.append(income - expense)

            fig.add_trace(
                go.Scatter(name='Net', x=list(months), y=net_by_month,
                           mode='lines+markers', line=dict(color='#6366f1', width=3),
                           marker=dict(size=8)),
                secondary_y=True
            )

            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=12),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom",
                            y=1.02, xanchor="right", x=1),
                barmode='group'
            )

            fig.update_yaxes(title_text="Amount ($)", secondary_y=False)
            fig.update_yaxes(title_text="Net ($)", secondary_y=True)

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 Import transactions to see cash flow analysis")

        st.markdown("---")

        # Spending Breakdown
        st.subheader("🎯 Spending Breakdown")

        if not transactions_df.empty:
            col_a, col_b = st.columns(2)

            with col_a:
                # Category pie chart
                category_spending = transactions_df[
                    (transactions_df['Date'] >= current_month_start) &
                    (transactions_df['Type'] == 'expense')
                ].groupby('Category')['Amount'].sum().abs()

                if not category_spending.empty:
                    fig = px.pie(
                        values=category_spending.values,
                        names=category_spending.index,
                        title="By Category",
                        hole=0.4,
                        color_discrete_sequence=px.colors.sequential.Purples_r
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        height=300,
                        showlegend=True,
                        legend=dict(font=dict(size=10))
                    )
                    st.plotly_chart(fig, use_container_width=True)

            with col_b:
                # Account breakdown
                account_spending = transactions_df[
                    (transactions_df['Date'] >= current_month_start) &
                    (transactions_df['Type'] == 'expense')
                ].groupby('Account')['Amount'].sum().abs()

                if not account_spending.empty:
                    fig = px.bar(
                        x=account_spending.index,
                        y=account_spending.values,
                        title="By Account",
                        color=account_spending.values,
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        height=300,
                        xaxis_title="",
                        yaxis_title="Amount ($)",
                        showlegend=False
                    )
                    st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")

        # Recent transactions
        st.subheader("📝 Recent Transactions")
        if not transactions_df.empty:
            recent = transactions_df.sort_values(
                'Date', ascending=False).head(10)

            for _, row in recent.iterrows():
                amount_color = "#10b981" if row['Type'] == 'income' else "#ef4444"
                amount_sign = "+" if row['Type'] == 'income' else "-"

                st.markdown(f"""
                <div style='background: var(--bg-tertiary); padding: 16px; border-radius: 12px; margin: 8px 0; 
                            border-left: 4px solid {amount_color}; transition: all 0.2s ease;'
                     onmouseover="this.style.transform='translateX(4px)'"
                     onmouseout="this.style.transform='translateX(0)'">
                    <div style='display: flex; justify-content: space-between; align-items: center;'>
                        <div>
                            <strong style='font-size: 16px;'>{row['Description']}</strong><br>
                            <span style='color: var(--text-secondary); font-size: 12px;'>
                                {row['Category']} • {row['Date'].strftime('%b %d, %Y')}
                            </span>
                        </div>
                        <div style='text-align: right;'>
                            <strong style='color: {amount_color}; font-size: 20px; font-weight: 700;'>
                                {amount_sign}${abs(row['Amount']):,.2f}
                            </strong><br>
                            <span style='color: var(--text-secondary); font-size: 11px;'>
                                {row.get('Account', 'N/A')}
                            </span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No transactions yet. Add some to get started!")

    with col_right:
        # Budget status
        st.subheader("💳 Budget Health")

        if not transactions_df.empty and not budgets_df.empty:
            current_spending = transactions_df[
                (transactions_df['Date'] >= current_month_start) &
                (transactions_df['Type'] == 'expense')
            ].groupby('Category')['Amount'].sum().abs()

            # Calculate overall budget health
            total_budget = budgets_df['Budget'].sum()
            total_spent = current_spending.sum()
            overall_health = (1 - (total_spent / total_budget)
                              ) * 100 if total_budget > 0 else 100

            health_color = "#10b981" if overall_health > 30 else "#f59e0b" if overall_health > 10 else "#ef4444"

            st.markdown(f"""
            <div style='background: linear-gradient(135deg, {health_color}22 0%, {health_color}11 100%);
                        border: 2px solid {health_color}; border-radius: 12px; padding: 16px; margin-bottom: 16px;'>
                <div style='text-align: center;'>
                    <h2 style='margin: 0; color: {health_color}; font-size: 48px;'>{overall_health:.0f}%</h2>
                    <p style='margin: 8px 0 0 0; color: var(--text-secondary);'>Budget Remaining</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

            for _, budget_row in budgets_df.iterrows():
                category = budget_row['Category']
                budget = budget_row['Budget']
                spent = current_spending.get(category, 0)
                pct = (spent / budget * 100) if budget > 0 else 0

                color = "#10b981" if pct < 70 else "#f59e0b" if pct < 90 else "#ef4444"

                st.markdown(f"""
                <div style='background: var(--bg-tertiary); padding: 14px; border-radius: 10px; margin: 10px 0;'>
                    <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                        <span style='font-weight: 600; font-size: 13px;'>{category}</span>
                        <span style='color: {color}; font-weight: 600; font-size: 13px;'>
                            ${spent:.0f} / ${budget:.0f}
                        </span>
                    </div>
                    <div style='background: var(--bg-secondary); height: 6px; border-radius: 3px; overflow: hidden;'>
                        <div style='background: {color}; height: 100%; width: {min(pct, 100):.1f}%; 
                                    transition: width 0.3s ease;'></div>
                    </div>
                    <div style='margin-top: 4px; display: flex; justify-content: space-between;'>
                        <span style='font-size: 10px; color: var(--text-secondary);'>{pct:.0f}% used</span>
                        <span style='font-size: 10px; color: var(--text-secondary);'>
                            ${budget - spent:.0f} left
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Set up budgets to track spending")

        st.markdown("---")

        # Goals Progress
        st.subheader("🎯 Goals")

        if not goals_df.empty:
            for _, goal in goals_df.head(3).iterrows():
                progress = (goal['Current_Amount'] / goal['Target_Amount']
                            * 100) if goal['Target_Amount'] > 0 else 0
                days_left = (goal['Target_Date'] - datetime.now()).days

                st.markdown(f"""
                <div class='glass-card' style='padding: 16px; margin: 12px 0;'>
                    <strong style='font-size: 14px;'>{goal['Goal_Name']}</strong><br>
                    <div style='margin: 8px 0;'>
                        <div style='background: var(--bg-secondary); height: 8px; border-radius: 4px; overflow: hidden;'>
                            <div style='background: linear-gradient(90deg, #6366f1, #8b5cf6); 
                                        height: 100%; width: {min(progress, 100):.1f}%;'></div>
                        </div>
                    </div>
                    <div style='display: flex; justify-content: space-between; font-size: 11px; color: var(--text-secondary);'>
                        <span>${goal['Current_Amount']:,.0f} / ${goal['Target_Amount']:,.0f}</span>
                        <span>{days_left} days left</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Set financial goals to track progress")

        st.markdown("---")

        # Quick insights
        st.subheader("💡 Smart Insights")
        insights = get_spending_insights(transactions_df, budgets_df)

        if insights:
            for insight in insights[:3]:
                st.markdown(f"""
                <div class='warning-card'>
                    <strong>⚠️ {insight['category']}</strong><br>
                    <span style='font-size: 13px;'>{insight['message']}</span>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class='success-card'>
                <strong>✅ Excellent!</strong><br>
                <span style='font-size: 13px;'>All spending within budget limits</span>
            </div>
            """, unsafe_allow_html=True)

        # Predicted spending alert
        if not transactions_df.empty:
            predicted = predict_end_of_month_spending(transactions_df)
            total_budget = budgets_df['Budget'].sum()

            if predicted > total_budget:
                st.markdown(f"""
                <div class='warning-card'>
                    <strong>📊 Spending Forecast</strong><br>
                    <span style='font-size: 13px;'>
                        Predicted month-end: ${predicted:,.0f}<br>
                        Over budget by: ${predicted - total_budget:,.0f}
                    </span>
                </div>
                """, unsafe_allow_html=True)

# ACCOUNTS PAGE
elif page == "💰 Accounts":
    st.title("Accounts")
    st.markdown("*Manage all your financial accounts*")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💼 Your Accounts")

        if not accounts_df.empty:
            total_balance = accounts_df['Balance'].sum()

            # Account type summary
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                checking = accounts_df[accounts_df['Account_Type']
                                       == 'Checking']['Balance'].sum()
                st.metric("Checking", f"${checking:,.2f}")
            with col_b:
                savings = accounts_df[accounts_df['Account_Type']
                                      == 'Savings']['Balance'].sum()
                st.metric("Savings", f"${savings:,.2f}")
            with col_c:
                investment = accounts_df[accounts_df['Account_Type']
                                         == 'Investment']['Balance'].sum()
                st.metric("Investment", f"${investment:,.2f}")

            st.markdown("---")

            # Display accounts
            for idx, account in accounts_df.iterrows():
                account_color = "#10b981" if account['Balance'] > 0 else "#ef4444"

                # Account type icon
                icon = "💳" if account['Account_Type'] == "Credit Card" else "🏦" if account[
                    'Account_Type'] == "Checking" else "💰" if account['Account_Type'] == "Savings" else "📈"

                with st.expander(f"{icon} {account['Account_Name']} - ${account['Balance']:,.2f}", expanded=False):
                    col_i, col_ii = st.columns(2)

                    with col_i:
                        st.markdown(f"""
                        **Type:** {account['Account_Type']}<br>
                        **Institution:** {account['Institution']}<br>
                        **Currency:** {account.get('Currency', 'USD')}<br>
                        **Interest Rate:** {account.get('Interest_Rate', 0)}%
                        """, unsafe_allow_html=True)

                    with col_ii:
                        new_balance = st.number_input("Update Balance", value=float(
                            account['Balance']), key=f"bal_{idx}")
                        if st.button("Update", key=f"update_acc_{idx}"):
                            accounts_df.at[idx, 'Balance'] = new_balance
                            save_accounts(accounts_df)
                            st.success("Balance updated!")
                            st.rerun()

                        if st.button("🗑️ Delete Account", key=f"del_acc_{idx}"):
                            accounts_df = accounts_df.drop(
                                idx).reset_index(drop=True)
                            save_accounts(accounts_df)
                            st.rerun()

            # Net worth chart
            st.markdown("---")
            st.subheader("📊 Account Distribution")

            fig = px.treemap(
                accounts_df,
                path=['Account_Type', 'Account_Name'],
                values='Balance',
                color='Balance',
                color_continuous_scale='RdYlGn',
                title="Account Hierarchy"
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No accounts added yet. Add your first account to get started!")

    with col2:
        st.subheader("➕ Add New Account")

        with st.form("add_account_form", clear_on_submit=True):
            account_name = st.text_input(
                "Account Nickname", placeholder="My Savings Account")
            account_type = st.selectbox(
                "Account Type", ["Checking", "Savings", "Credit Card", "Investment", "Cash"])
            institution = st.text_input(
                "Bank/Institution", placeholder="Chase, Vanguard, etc.")

            col_a, col_b = st.columns(2)
            with col_a:
                balance = st.number_input(
                    "Current Balance ($)", value=0.0, step=100.0)
            with col_b:
                currency = st.selectbox(
                    "Currency", ["USD", "EUR", "GBP", "CAD", "AUD"])

            interest_rate = st.number_input(
                "Interest Rate (%)", value=0.0, step=0.1, help="Annual interest rate if applicable")

            if st.form_submit_button("➕ Add Account", type="primary", use_container_width=True):
                if account_name and institution:
                    new_account = pd.DataFrame([{
                        'Account_Name': account_name,
                        'Account_Type': account_type,
                        'Balance': balance,
                        'Institution': institution,
                        'Currency': currency,
                        'Interest_Rate': interest_rate
                    }])

                    if accounts_df.empty:
                        accounts_df = new_account
                    else:
                        accounts_df = pd.concat(
                            [accounts_df, new_account], ignore_index=True)

                    save_accounts(accounts_df)
                    st.success("✅ Account added successfully!")
                    st.rerun()
                else:
                    st.error("Please fill in all required fields")

        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        <div class='insight-card'>
        • Link all your accounts for complete overview<br>
        • Update balances regularly<br>
        • Track interest rates for optimization<br>
        • Separate emergency fund accounts
        </div>
        """, unsafe_allow_html=True)

# Continue in next message due to length...

    # TRANSACTIONS PAGE
elif page == "📊 Transactions":
    st.title("Transactions")
    st.markdown("*Track and manage your transactions*")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Trends", "🎯 Categories", "💳 Accounts", "📅 Calendar", "🔗 Linked Payments"])


    # TRANSACTIONS PAGE - Import Tab (Update this section)
    # with tab1:
    #     st.subheader("Import Bank Statement")

    #     col1, col2 = st.columns([2, 1])

    #     with col1:
    #         st.markdown("""
    #         **📄 CSV Format Required:**
    #         - Date (YYYY-MM-DD or MM/DD/YYYY)
    #         - Description
    #         - Amount (negative for expenses, positive for income)
    #         - Category (optional - AI will categorize)
    #         - Account (optional)
    #         """)

    #         uploaded_file = st.file_uploader("📁 Upload CSV file", type=['csv'])

    #         if uploaded_file:
    #             try:
    #                 new_transactions = pd.read_csv(uploaded_file)

    #                 # Show preview
    #                 st.success(
    #                     f"✅ Loaded {len(new_transactions)} transactions")
    #                 st.dataframe(new_transactions.head(
    #                     10), use_container_width=True)

    #                 # Parse dates
    #                 new_transactions['Date'] = pd.to_datetime(
    #                     new_transactions['Date'], errors='coerce')

    #                 # Add missing columns
    #                 if 'Type' not in new_transactions.columns:
    #                     new_transactions['Type'] = new_transactions['Amount'].apply(
    #                         lambda x: 'income' if x > 0 else 'expense'
    #                     )

    #                 if 'Category' not in new_transactions.columns:
    #                     if AI_ENABLED:
    #                         st.info(
    #                             "🤖 AI is categorizing all transactions in one batch...")
    #                         with st.spinner("Processing..."):
    #                             try:
    #                                 # Convert to list of dicts for batch processing
    #                                 transactions_list = new_transactions[[
    #                                     'Description', 'Amount']].to_dict('records')

    #                                 # Single AI call for all transactions
    #                                 categories = ai_categorize_transactions_batch(
    #                                     transactions_list)
    #                                 new_transactions['Category'] = categories

    #                                 st.success(
    #                                     f"✅ Categorized {len(categories)} transactions!")
    #                             except Exception as e:
    #                                 st.warning(
    #                                     f"⚠️ AI categorization failed: {e}. Using simple categorization...")
    #                                 new_transactions['Category'] = new_transactions.apply(
    #                                     lambda row: ai_categorize_transaction_simple(
    #                                         row['Description'], row['Amount']),
    #                                     axis=1
    #                                 )
    #                     else:
    #                         st.info("🔧 Using rule-based categorization...")
    #                         new_transactions['Category'] = new_transactions.apply(
    #                             lambda row: ai_categorize_transaction_simple(
    #                                 row['Description'], row['Amount']),
    #                             axis=1
    #                         )

    #                 if 'Account' not in new_transactions.columns:
    #                     default_account = accounts_df['Account_Name'].iloc[0] if not accounts_df.empty else 'Imported'
    #                     new_transactions['Account'] = default_account

    #                 if 'Source' not in new_transactions.columns:
    #                     new_transactions['Source'] = uploaded_file.name

    #                 if 'Tags' not in new_transactions.columns:
    #                     new_transactions['Tags'] = ''

    #                 # Show breakdown
    #                 st.subheader("📊 Import Summary")

    #                 col_a, col_b = st.columns(2)

    #                 with col_a:
    #                     category_summary = new_transactions.groupby('Category').agg({
    #                         'Amount': ['sum', 'count']
    #                     }).reset_index()
    #                     category_summary.columns = [
    #                         'Category', 'Total', 'Count']
    #                     category_summary['Total'] = category_summary['Total'].apply(
    #                         lambda x: f"${abs(x):,.2f}")
    #                     st.dataframe(category_summary,
    #                                  use_container_width=True, hide_index=True)

    #                 with col_b:
    #                     expense_transactions = new_transactions[new_transactions['Type'] == 'expense']
    #                     if not expense_transactions.empty:
    #                         fig = px.pie(
    #                             expense_transactions,
    #                             names='Category',
    #                             values=expense_transactions['Amount'].abs(),
    #                             title="Expense Distribution",
    #                             color_discrete_sequence=px.colors.sequential.Purples_r
    #                         )
    #                         fig.update_layout(
    #                             plot_bgcolor='rgba(0,0,0,0)',
    #                             paper_bgcolor='rgba(0,0,0,0)',
    #                             font=dict(color='white'),
    #                             height=300
    #                         )
    #                         st.plotly_chart(fig, use_container_width=True)

    #                 if st.button("✅ Confirm Import", type="primary", use_container_width=True):
    #                     if transactions_df.empty:
    #                         transactions_df = new_transactions
    #                     else:
    #                         transactions_df = pd.concat(
    #                             [transactions_df, new_transactions], ignore_index=True)

    #                     save_transactions(transactions_df)
    #                     st.success(
    #                         f"🎉 Successfully imported {len(new_transactions)} transactions!")
    #                     st.balloons()
    #                     st.rerun()

    #             except Exception as e:
    #                 st.error(f"❌ Error processing file: {str(e)}")
    #                 st.exception(e)

    #     with col2:
    #         st.markdown("### 💡 Tips")
    #         st.markdown("""
    #         <div class='insight-card'>
    #         • Most bank CSV formats supported<br>
    #         • AI categorizes all transactions at once<br>
    #         • Review before importing<br>
    #         • Negative = expense, Positive = income<br>
    #         • Large imports may take a moment
    #         </div>
    #         """, unsafe_allow_html=True)

    #         # Sample CSV download
    #         sample_csv = pd.DataFrame({
    #             'Date': ['2025-01-15', '2025-01-16', '2025-01-17'],
    #             'Description': ['Grocery Store', 'Gas Station', 'Restaurant'],
    #             'Amount': [-85.50, -45.00, -62.30],
    #             'Category': ['Food & Dining', 'Transportation', 'Food & Dining'],
    #             'Account': ['Checking', 'Credit Card', 'Credit Card']
    #         })

    #         csv = sample_csv.to_csv(index=False)
    #         st.download_button(
    #             label="📥 Download Sample CSV",
    #             data=csv,
    #             file_name="sample_transactions.csv",
    #             mime="text/csv",
    #             use_container_width=True
    #         )



    # ###

    # TRANSACTIONS PAGE - Update the Import Tab
    with tab1:
        st.subheader("Import Bank Statement")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # File type selection - THIS IS THE NEW PART
            import_format = st.radio(
                "Select file format:",
                ["QFX/OFX (Quicken)", "CSV (Comma-Separated)"],
                horizontal=True
            )
            
            if import_format == "QFX/OFX (Quicken)":
                st.markdown("""
                **📄 QFX/OFX Format:**
                - Direct import from your bank
                - Automatic account detection
                - Transaction IDs for duplicate prevention
                - Most banks support this format
                """)
                file_types = ['qfx', 'ofx']
            else:
                st.markdown("""
                **📄 CSV Format Required:**
                - Date (YYYY-MM-DD or MM/DD/YYYY)
                - Description
                - Amount (negative for expenses, positive for income)
                - Category (optional - AI will categorize)
                - Account (optional - will use selected account below)
                """)
                file_types = ['csv']
            
            # Account selection for import
            st.markdown("---")
            col_import1, col_import2 = st.columns(2)
            
            with col_import1:
                if not accounts_df.empty:
                    import_account = st.selectbox(
                        "📁 Import to Account",
                        accounts_df['Account_Name'].tolist(),
                        help="Select which account these transactions belong to"
                    )
                else:
                    import_account = st.text_input("Account Name", value="Imported Account", help="No accounts found. Enter account name:")
                    if st.button("Create Account", key="create_import_account"):
                        new_account = pd.DataFrame([{
                            'Account_Name': import_account,
                            'Account_Type': 'Checking',
                            'Balance': 0.0,
                            'Institution': 'Unknown',
                            'Currency': 'USD',
                            'Interest_Rate': 0.0
                        }])
                        if accounts_df.empty:
                            accounts_df = new_account
                        else:
                            accounts_df = pd.concat([accounts_df, new_account], ignore_index=True)
                        save_accounts(accounts_df)
                        st.success(f"✅ Account '{import_account}' created!")
                        st.rerun()
            
            with col_import2:
                link_recurring = st.checkbox("🔗 Auto-link recurring", value=True, help="Automatically detect and link to existing recurring transactions")
                link_loans = st.checkbox("🏦 Auto-link loans", value=True, help="Automatically detect and link to existing loans")
            
            st.markdown("---")
            
            # File uploader - UPDATED TO SUPPORT BOTH FORMATS
            uploaded_file = st.file_uploader(
                f"📁 Upload {import_format.split()[0]} file",
                type=file_types,
                help=f"Upload your bank statement in {import_format} format"
            )
            
            if uploaded_file:
                try:
                    # Parse file based on format - NEW LOGIC
                    if import_format == "QFX/OFX (Quicken)":
                        if not OFX_AVAILABLE:
                            st.error("❌ QFX parsing not available. Install with: pip install ofxparse")
                            st.stop()
                        else:
                            with st.spinner("Parsing QFX file..."):
                                new_transactions, detected_account = parse_qfx_file(uploaded_file)
                                
                                if detected_account:
                                    st.info(f"🏦 Detected account: {detected_account}")
                    else:
                        with st.spinner("Parsing CSV file..."):
                            new_transactions, detected_account = parse_csv_file(uploaded_file)
                    
                    # Show preview
                    st.success(f"✅ Loaded {len(new_transactions)} transactions")
                    
                    # Remove duplicates for QFX files
                    if 'Transaction_ID' in new_transactions.columns:
                        if not transactions_df.empty and 'Transaction_ID' in transactions_df.columns:
                            existing_ids = transactions_df['Transaction_ID'].tolist()
                            original_count = len(new_transactions)
                            new_transactions = new_transactions[~new_transactions['Transaction_ID'].isin(existing_ids)]
                            if len(new_transactions) < original_count:
                                st.info(f"✅ Filtered out {original_count - len(new_transactions)} duplicate transactions.")
                    
                    st.dataframe(new_transactions.head(10), use_container_width=True)
                    
                    # Parse dates
                    new_transactions['Date'] = pd.to_datetime(new_transactions['Date'], errors='coerce')
                    
                    # Add missing columns
                    if 'Type' not in new_transactions.columns:
                        new_transactions['Type'] = new_transactions['Amount'].apply(
                            lambda x: 'income' if x > 0 else 'expense'
                        )
                    
                    if 'Category' not in new_transactions.columns:
                        if AI_ENABLED:
                            st.info("🤖 AI is categorizing all transactions in one batch...")
                            with st.spinner("Processing..."):
                                try:
                                    transactions_list = new_transactions[['Description', 'Amount']].to_dict('records')
                                    categories = ai_categorize_transactions_batch(transactions_list)
                                    new_transactions['Category'] = categories
                                    st.success(f"✅ Categorized {len(categories)} transactions!")
                                except Exception as e:
                                    st.warning(f"⚠️ AI categorization failed: {e}. Using simple categorization...")
                                    new_transactions['Category'] = new_transactions.apply(
                                        lambda row: ai_categorize_transaction_simple(row['Description'], row['Amount']),
                                        axis=1
                                    )
                        else:
                            st.info("🔧 Using rule-based categorization...")
                            new_transactions['Category'] = new_transactions.apply(
                                lambda row: ai_categorize_transaction_simple(row['Description'], row['Amount']),
                                axis=1
                            )
                    
                    # Set account for all transactions
                    if 'Account' not in new_transactions.columns or new_transactions['Account'].isna().all():
                        new_transactions['Account'] = import_account
                    
                    if 'Source' not in new_transactions.columns:
                        new_transactions['Source'] = uploaded_file.name
                    
                    if 'Tags' not in new_transactions.columns:
                        new_transactions['Tags'] = ''
                    
                    # Initialize linking columns
                    if 'Linked_Recurring_ID' not in new_transactions.columns:
                        new_transactions['Linked_Recurring_ID'] = ''
                    if 'Linked_Loan_ID' not in new_transactions.columns:
                        new_transactions['Linked_Loan_ID'] = ''
                    if 'Transaction_ID' not in new_transactions.columns:
                        new_transactions['Transaction_ID'] = ''
                    
                    # Auto-link to recurring payments
                    linked_recurring_count = 0
                    linked_loan_count = 0
                    
                    if link_recurring and not recurring_df.empty:
                        st.info("🔗 Linking to recurring payments...")
                        
                        for idx, trans in new_transactions.iterrows():
                            for rec_idx, rec in recurring_df.iterrows():
                                # Check if description matches (fuzzy match)
                                desc_match = (
                                    rec['Description'].lower() in trans['Description'].lower() or 
                                    trans['Description'].lower() in rec['Description'].lower()
                                )
                                
                                # Check if amount is similar (within 5%)
                                if rec['Amount'] != 0:
                                    amount_match = abs(abs(trans['Amount']) - abs(rec['Amount'])) / abs(rec['Amount']) < 0.05
                                else:
                                    amount_match = False
                                
                                if desc_match and amount_match:
                                    new_transactions.at[idx, 'Linked_Recurring_ID'] = str(rec_idx)
                                    current_tags = str(new_transactions.at[idx, 'Tags'])
                                    new_tags = f"{current_tags},recurring,subscription" if current_tags else "recurring,subscription"
                                    new_transactions.at[idx, 'Tags'] = new_tags.strip(',')
                                    linked_recurring_count += 1
                                    break
                        
                        if linked_recurring_count > 0:
                            st.success(f"✅ Linked {linked_recurring_count} transactions to recurring payments!")
                    
                    # Auto-link to loan payments
                    if link_loans and not loans_df.empty:
                        st.info("🏦 Linking to loan payments...")
                        
                        for idx, trans in new_transactions.iterrows():
                            for loan_idx, loan in loans_df.iterrows():
                                # Check if description matches loan
                                desc_match = loan['Description'].lower() in trans['Description'].lower()
                                
                                # Check if amount matches monthly payment (within 5%)
                                if loan['Monthly_Payment'] != 0:
                                    amount_match = abs(abs(trans['Amount']) - loan['Monthly_Payment']) / loan['Monthly_Payment'] < 0.05
                                else:
                                    amount_match = False
                                
                                if desc_match and amount_match:
                                    new_transactions.at[idx, 'Linked_Loan_ID'] = str(loan_idx)
                                    current_tags = str(new_transactions.at[idx, 'Tags'])
                                    new_tags = f"{current_tags},loan-payment" if current_tags else "loan-payment"
                                    new_transactions.at[idx, 'Tags'] = new_tags.strip(',')
                                    linked_loan_count += 1
                                    break
                        
                        if linked_loan_count > 0:
                            st.success(f"✅ Linked {linked_loan_count} transactions to loan payments!")
                    
                    # Show breakdown
                    st.subheader("📊 Import Summary")
                    
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        category_summary = new_transactions.groupby('Category').agg({
                            'Amount': ['sum', 'count']
                        }).reset_index()
                        category_summary.columns = ['Category', 'Total', 'Count']
                        category_summary['Total'] = category_summary['Total'].apply(lambda x: f"${abs(x):,.2f}")
                        st.dataframe(category_summary, use_container_width=True, hide_index=True)
                        
                        # Show linked items
                        st.markdown(f"""
    <div class='success-card' style='margin-top: 12px;'>
        <strong>🔗 Auto-Linked:</strong><br>
        • {linked_recurring_count} recurring payments<br>
        • {linked_loan_count} loan payments
    </div>
    """, unsafe_allow_html=True)
                    
                    with col_b:
                        expense_transactions = new_transactions[new_transactions['Type'] == 'expense']
                        if not expense_transactions.empty:
                            fig = px.pie(
                                expense_transactions,
                                names='Category',
                                values=expense_transactions['Amount'].abs(),
                                title="Expense Distribution",
                                color_discrete_sequence=px.colors.sequential.Purples_r
                            )
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'),
                                height=300
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    if st.button("✅ Confirm Import", type="primary", use_container_width=True):
                        # Update account balances
                        if not accounts_df.empty:
                            account_idx = accounts_df[accounts_df['Account_Name'] == import_account].index
                            if not account_idx.empty:
                                total_change = new_transactions['Amount'].sum()
                                accounts_df.at[account_idx[0], 'Balance'] += total_change
                                save_accounts(accounts_df)
                        
                        # Update loan balances for linked transactions
                        if linked_loan_count > 0:
                            for idx, trans in new_transactions.iterrows():
                                loan_id = trans.get('Linked_Loan_ID', '')
                                if loan_id and loan_id != '':
                                    try:
                                        loan_idx = int(float(loan_id))
                                        if loan_idx < len(loans_df):
                                            current_remaining = loans_df.at[loan_idx, 'Remaining']
                                            loans_df.at[loan_idx, 'Remaining'] = max(0, current_remaining - abs(trans['Amount']))
                                    except:
                                        pass
                            save_loans(loans_df)
                        
                        if transactions_df.empty:
                            transactions_df = new_transactions
                        else:
                            transactions_df = pd.concat([transactions_df, new_transactions], ignore_index=True)
                        
                        save_transactions(transactions_df)
                        st.success(f"🎉 Successfully imported {len(new_transactions)} transactions to {import_account}!")
                        st.balloons()
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
                    st.exception(e)
        
        with col2:
            st.markdown("### 💡 Import Tips")
            st.markdown(f"""
    <div class='insight-card'>
    • <strong>Format:</strong> {import_format}<br>
    • <strong>Account:</strong> {import_account if 'import_account' in locals() else 'Select account'}<br>
    • Balance updates automatically<br>
    • Duplicates filtered (QFX)<br>
    • Auto-links recurring & loans<br>
    • AI categorizes transactions
    </div>
    """, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("### 📥 Download Sample")
            
            # Sample CSV download
            sample_csv = pd.DataFrame({
                'Date': ['2025-01-15', '2025-01-16', '2025-01-17'],
                'Description': ['Grocery Store', 'Gas Station', 'Restaurant'],
                'Amount': [-85.50, -45.00, -62.30],
            })
            
            csv = sample_csv.to_csv(index=False)
            st.download_button(
                label="📥 Sample CSV",
                data=csv,
                file_name="sample_transactions.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.markdown("---")
            st.markdown("### 🏦 Getting QFX Files")
            st.markdown("""
    <div class='insight-card' style='font-size: 12px;'>
    1. Log into your bank<br>
    2. Go to account transactions<br>
    3. Look for "Export" or "Download"<br>
    4. Select QFX/OFX/Quicken format<br>
    5. Upload here!<br><br>
    <strong>Supported banks:</strong> Most major banks including Chase, Bank of America, Wells Fargo, Citi, etc.
    </div>
    """, unsafe_allow_html=True)


    with tab2:
        st.subheader("Add Transaction Manually")

        col1, col2 = st.columns([2, 1])

        with col1:
            with st.form("manual_transaction", clear_on_submit=True):
                date = st.date_input("📅 Date", value=datetime.now())
                description = st.text_input(
                    "📝 Description", placeholder="e.g., Starbucks Coffee")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    amount = st.number_input(
                        "💵 Amount ($)", min_value=0.01, step=0.01)
                with col_b:
                    trans_type = st.selectbox("Type", ["expense", "income"])
                with col_c:
                    category = st.selectbox("Category",
                                            budgets_df['Category'].tolist() + ['Income', 'Investment', 'Other'])

                account = st.selectbox("💳 Account",
                                       accounts_df['Account_Name'].tolist() if not accounts_df.empty else ['Manual Entry'])

                tags = st.text_input(
                    "🏷️ Tags (comma separated)", placeholder="food, coffee, work")
                notes = st.text_area("📋 Notes (optional)",
                                     placeholder="Additional details...")

                if st.form_submit_button("➕ Add Transaction", type="primary", use_container_width=True):
                    if description and amount > 0:
                        new_trans = pd.DataFrame([{
                            'Date': pd.to_datetime(date),
                            'Description': description,
                            'Amount': amount if trans_type == 'income' else -abs(amount),
                            'Category': category,
                            'Type': trans_type,
                            'Account': account,
                            'Source': 'Manual',
                            'Tags': tags
                        }])

                        if transactions_df.empty:
                            transactions_df = new_trans
                        else:
                            transactions_df = pd.concat(
                                [transactions_df, new_trans], ignore_index=True)

                        save_transactions(transactions_df)
                        st.success("✅ Transaction added!")
                        st.rerun()
                    else:
                        st.error("Please fill in all required fields")

        with col2:
            st.markdown("### 🎯 Quick Add")

            # Quick expense buttons
            st.markdown("**Common Expenses:**")
            quick_expenses = {
                "☕ Coffee": 5,
                "🍔 Lunch": 15,
                "⛽ Gas": 50,
                "🎬 Movie": 12,
                "🚕 Uber": 20
            }

            for label, amount in quick_expenses.items():
                if st.button(label, use_container_width=True):
                    quick_trans = pd.DataFrame([{
                        'Date': pd.to_datetime(datetime.now()),
                        'Description': label,
                        'Amount': -amount,
                        'Category': ai_categorize_transaction(label, -amount),
                        'Type': 'expense',
                        'Account': accounts_df['Account_Name'].iloc[0] if not accounts_df.empty else 'Manual',
                        'Source': 'Quick Add',
                        'Tags': ''
                    }])

                    if transactions_df.empty:
                        transactions_df = quick_trans
                    else:
                        transactions_df = pd.concat(
                            [transactions_df, quick_trans], ignore_index=True)

                    save_transactions(transactions_df)
                    st.success(f"Added {label} - ${amount}")
                    st.rerun()

    with tab3:
        st.subheader("All Transactions")

        if not transactions_df.empty:
            # Advanced filters
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                filter_type = st.selectbox(
                    "Type", ["All", "expense", "income"])
            with col2:
                filter_category = st.selectbox(
                    "Category", ["All"] + sorted(transactions_df['Category'].unique().tolist()))
            with col3:
                filter_account = st.selectbox(
                    "Account", ["All"] + sorted(transactions_df['Account'].unique().tolist()))
            with col4:
                date_range = st.selectbox("Period", [
                                          "All Time", "This Month", "Last Month", "Last 3 Months", "Last 6 Months", "This Year"])

            # Apply filters
            filtered_df = transactions_df.copy()

            if filter_type != "All":
                filtered_df = filtered_df[filtered_df['Type'] == filter_type]
            if filter_category != "All":
                filtered_df = filtered_df[filtered_df['Category']
                                          == filter_category]
            if filter_account != "All":
                filtered_df = filtered_df[filtered_df['Account']
                                          == filter_account]

            # Date filter
            today = datetime.now()
            if date_range == "This Month":
                filtered_df = filtered_df[filtered_df['Date']
                                          >= today.replace(day=1)]
            elif date_range == "Last Month":
                last_month = (today.replace(day=1) -
                              timedelta(days=1)).replace(day=1)
                filtered_df = filtered_df[
                    (filtered_df['Date'] >= last_month) &
                    (filtered_df['Date'] < today.replace(day=1))
                ]
            elif date_range == "Last 3 Months":
                filtered_df = filtered_df[filtered_df['Date']
                                          >= today - timedelta(days=90)]
            elif date_range == "Last 6 Months":
                filtered_df = filtered_df[filtered_df['Date']
                                          >= today - timedelta(days=180)]
            elif date_range == "This Year":
                filtered_df = filtered_df[filtered_df['Date']
                                          >= today.replace(month=1, day=1)]

            # Summary stats
            col_a, col_b, col_c, col_d = st.columns(4)
            with col_a:
                st.metric("Total Transactions", len(filtered_df))
            with col_b:
                total_in = filtered_df[filtered_df['Type']
                                       == 'income']['Amount'].sum()
                st.metric("Total Income", f"${total_in:,.2f}")
            with col_c:
                total_out = abs(
                    filtered_df[filtered_df['Type'] == 'expense']['Amount'].sum())
                st.metric("Total Expenses", f"${total_out:,.2f}")
            with col_d:
                net = total_in - total_out
                st.metric("Net", f"${net:,.2f}", delta=f"{net:+,.2f}")

            st.markdown("---")

            # Search
            search = st.text_input(
                "🔍 Search transactions", placeholder="Search by description...")
            if search:
                filtered_df = filtered_df[filtered_df['Description'].str.contains(
                    search, case=False, na=False)]

            st.markdown(f"**Showing {len(filtered_df)} transactions**")

            # Bulk actions
            col_bulk1, col_bulk2, col_bulk3 = st.columns([2, 2, 2])
            with col_bulk1:
                if st.button("✏️ Enable Edit Mode", use_container_width=True):
                    st.session_state.edit_mode = not st.session_state.get(
                        'edit_mode', False)
            with col_bulk2:
                bulk_category = st.selectbox("Bulk change category", [
                                             ""] + budgets_df['Category'].tolist(), key="bulk_cat")
            with col_bulk3:
                if st.button("Apply to filtered", use_container_width=True) and bulk_category:
                    for idx in filtered_df.index:
                        transactions_df.at[idx, 'Category'] = bulk_category
                    save_transactions(transactions_df)
                    st.success(f"Updated {len(filtered_df)} transactions!")
                    st.rerun()

            st.markdown("---")

            # Display transactions with edit capability
            display_df = filtered_df.sort_values(
                'Date', ascending=False).copy()

            # Add edit mode
            edit_mode = st.session_state.get('edit_mode', False)

            if edit_mode:
                st.info("✏️ **Edit Mode Active** - Click on any transaction to edit")

            # Display each transaction as an editable card
            for idx, row in display_df.iterrows():
                amount_color = "#10b981" if row['Type'] == 'income' else "#ef4444"
                amount_sign = "+" if row['Type'] == 'income' else "-"

                with st.expander(
                    f"{row['Date'].strftime('%Y-%m-%d')} • {row['Description']} • {amount_sign}${abs(row['Amount']):,.2f}",
                    expanded=False
                ):
                    if edit_mode:
                        # Edit form
                        with st.form(key=f"edit_trans_{idx}"):
                            col_e1, col_e2 = st.columns(2)

                            with col_e1:
                                new_description = st.text_input(
                                    "Description", value=row['Description'])
                                new_category = st.selectbox(
                                    "Category",
                                    budgets_df['Category'].tolist(
                                    ) + ['Income', 'Investment', 'Other'],
                                    index=(budgets_df['Category'].tolist(
                                    ) + ['Income', 'Investment', 'Other']).index(row['Category'])
                                    if row['Category'] in (budgets_df['Category'].tolist() + ['Income', 'Investment', 'Other']) else 0
                                )
                                new_date = st.date_input(
                                    "Date", value=row['Date'].date())

                            with col_e2:
                                new_amount = st.number_input(
                                    "Amount", value=float(abs(row['Amount'])), step=0.01)
                                new_account = st.selectbox(
                                    "Account",
                                    accounts_df['Account_Name'].tolist() if not accounts_df.empty else [
                                        'Manual'],
                                    index=accounts_df['Account_Name'].tolist().index(
                                        row['Account'])
                                    if not accounts_df.empty and row['Account'] in accounts_df['Account_Name'].tolist() else 0
                                )
                                new_tags = st.text_input(
                                    "Tags (comma-separated)", value=row.get('Tags', ''))

                            col_save, col_delete = st.columns(2)

                            with col_save:
                                if st.form_submit_button("💾 Save Changes", use_container_width=True):
                                    transactions_df.at[idx,
                                                       'Description'] = new_description
                                    transactions_df.at[idx,
                                                       'Category'] = new_category
                                    transactions_df.at[idx, 'Date'] = pd.to_datetime(
                                        new_date)
                                    transactions_df.at[idx, 'Amount'] = new_amount if row['Type'] == 'income' else -abs(
                                        new_amount)
                                    transactions_df.at[idx,
                                                       'Account'] = new_account
                                    transactions_df.at[idx, 'Tags'] = new_tags
                                    save_transactions(transactions_df)
                                    st.success("✅ Transaction updated!")
                                    st.rerun()

                            with col_delete:
                                if st.form_submit_button("🗑️ Delete", type="secondary", use_container_width=True):
                                    transactions_df = transactions_df.drop(
                                        idx).reset_index(drop=True)
                                    save_transactions(transactions_df)
                                    st.success("🗑️ Transaction deleted!")
                                    st.rerun()
                    else:
                        col_v1, col_v2 = st.columns(2)
                    
                        with col_v1:
                            st.markdown(f"""
                            **Description:** {row['Description']}<br>
                            **Category:** {row['Category']}<br>
                            **Date:** {row['Date'].strftime('%B %d, %Y')}<br>
                            **Tags:** {row.get('Tags', 'None')}
                            """, unsafe_allow_html=True)
                        
                        with col_v2:
                            st.markdown(f"""
                            **Amount:** <span style='color: {amount_color}; font-size: 20px; font-weight: bold;'>{amount_sign}${abs(row['Amount']):,.2f}</span><br>
                            **Type:** {row['Type'].capitalize()}<br>
                            **Account:** {row['Account']}<br>
                            **Source:** {row.get('Source', 'N/A')}
                            """, unsafe_allow_html=True)
                        
                        # Quick edit buttons
                        st.markdown("---")
                        col_q1, col_q2, col_q3 = st.columns(3)
                        
                        with col_q1:
                            # Quick category change
                            quick_cat = st.selectbox(
                                "Quick change category",
                                [""] + budgets_df['Category'].tolist(),
                                key=f"quick_cat_{idx}"
                            )
                            if quick_cat:
                                transactions_df.at[idx, 'Category'] = quick_cat
                                save_transactions(transactions_df)
                                st.rerun()
                        
                        with col_q2:
                            # Quick tag add
                            quick_tag = st.text_input("Add tag", key=f"quick_tag_{idx}", placeholder="vacation, work, etc.")
                            if st.button("+ Add", key=f"add_tag_{idx}"):
                                if quick_tag:
                                    current_tags = row.get('Tags', '')
                                    new_tags = f"{current_tags},{quick_tag}" if current_tags else quick_tag
                                    transactions_df.at[idx, 'Tags'] = new_tags
                                    save_transactions(transactions_df)
                                    st.success(f"Added tag: {quick_tag}")
                                    st.rerun()
                        
                        with col_q3:
                            if st.button("🗑️ Delete", key=f"del_{idx}", use_container_width=True):
                                transactions_df = transactions_df.drop(idx).reset_index(drop=True)
                                save_transactions(transactions_df)
                                st.rerun()
                        # Add linking section
                        st.markdown("---")
                        st.markdown("### 🔗 Link Payment")
                        
                        col_link1, col_link2 = st.columns(2)
                        current_recurring_link = str(row.get('Linked_Recurring_ID', ''))
                        current_loan_link = str(row.get('Linked_Loan_ID', ''))
                        
                        with col_link1:
                            st.markdown("**🔄 Recurring Payment**")
                            
                            if not recurring_df.empty:
                                
                                # Build recurring options
                                recurring_options = {"None": -1}
                                for rec_idx, rec in recurring_df.iterrows():
                                    option_text = f"{rec['Description']} (${abs(rec['Amount']):.2f})"
                                    recurring_options[option_text] = rec_idx
                                
                                # Find current selection
                                current_selection = "None"
                                if current_recurring_link and current_recurring_link != '' and current_recurring_link != 'nan':
                                    try:
                                        link_idx = int(float(current_recurring_link))
                                        for opt_text, opt_idx in recurring_options.items():
                                            if opt_idx == link_idx:
                                                current_selection = opt_text
                                                break
                                    except:
                                        pass
                                
                                selected_recurring = st.selectbox(
                                    "Select recurring payment:",
                                    list(recurring_options.keys()),
                                    index=list(recurring_options.keys()).index(current_selection),
                                    key=f"link_rec_{idx}"
                                )
                                
                                if st.button("🔗 Link", key=f"link_rec_btn_{idx}", use_container_width=True):
                                    if selected_recurring == "None":
                                        transactions_df.at[idx, 'Linked_Recurring_ID'] = ''
                                        # Remove recurring tags
                                        tags = str(row.get('Tags', ''))
                                        tag_list = [t.strip() for t in tags.split(',') if t.strip() not in ['recurring', 'subscription']]
                                        transactions_df.at[idx, 'Tags'] = ','.join(tag_list)
                                    else:
                                        rec_idx = recurring_options[selected_recurring]
                                        transactions_df.at[idx, 'Linked_Recurring_ID'] = str(rec_idx)
                                        # Add recurring tag
                                        tags = str(row.get('Tags', ''))
                                        tag_list = [t.strip() for t in tags.split(',') if t.strip()]
                                        if 'recurring' not in tag_list:
                                            tag_list.append('recurring')
                                        transactions_df.at[idx, 'Tags'] = ','.join(tag_list)
                                    
                                    save_transactions(transactions_df)
                                    st.success("✅ Updated!")
                                    st.rerun()
                            else:
                                st.info("No recurring payments available. Add one first!")
                        
                        with col_link2:
                            st.markdown("**🏦 Loan Payment**")
                            
                            if not loans_df.empty:
                                
                                
                                # Build loan options
                                loan_options = {"None": -1}
                                for loan_idx, loan in loans_df.iterrows():
                                    option_text = f"{loan['Description']} (${loan['Monthly_Payment']:.2f}/mo)"
                                    loan_options[option_text] = loan_idx
                                
                                # Find current selection
                                current_selection = "None"
                                if current_loan_link and current_loan_link != '' and current_loan_link != 'nan':
                                    try:
                                        link_idx = int(float(current_loan_link))
                                        for opt_text, opt_idx in loan_options.items():
                                            if opt_idx == link_idx:
                                                current_selection = opt_text
                                                break
                                    except:
                                        pass
                                
                                selected_loan = st.selectbox(
                                    "Select loan:",
                                    list(loan_options.keys()),
                                    index=list(loan_options.keys()).index(current_selection),
                                    key=f"link_loan_{idx}"
                                )
                                
                                if st.button("🔗 Link", key=f"link_loan_btn_{idx}", use_container_width=True):
                                    if selected_loan == "None":
                                        transactions_df.at[idx, 'Linked_Loan_ID'] = ''
                                        # Remove loan-payment tag
                                        tags = str(row.get('Tags', ''))
                                        tag_list = [t.strip() for t in tags.split(',') if t.strip() != 'loan-payment']
                                        transactions_df.at[idx, 'Tags'] = ','.join(tag_list)
                                    else:
                                        loan_idx = loan_options[selected_loan]
                                        transactions_df.at[idx, 'Linked_Loan_ID'] = str(loan_idx)
                                        # Add loan-payment tag
                                        tags = str(row.get('Tags', ''))
                                        tag_list = [t.strip() for t in tags.split(',') if t.strip()]
                                        if 'loan-payment' not in tag_list:
                                            tag_list.append('loan-payment')
                                        transactions_df.at[idx, 'Tags'] = ','.join(tag_list)
                                        
                                        # Update loan remaining balance
                                        if loan_idx < len(loans_df):
                                            current_remaining = loans_df.at[loan_idx, 'Remaining']
                                            loans_df.at[loan_idx, 'Remaining'] = max(0, current_remaining - abs(row['Amount']))
                                            save_loans(loans_df)
                                    
                                    save_transactions(transactions_df)
                                    st.success("✅ Updated!")
                                    st.rerun()
                            else:
                                st.info("No loans available. Add one first!")
                        
                        # Show current links
                        if current_recurring_link and current_recurring_link not in ['', 'nan']:
                            try:
                                link_idx = int(float(current_recurring_link))
                                if link_idx < len(recurring_df):
                                    rec = recurring_df.iloc[link_idx]
                                    st.info(f"🔗 Linked to: {rec['Description']}")
                            except:
                                pass
                        
                        if current_loan_link and current_loan_link not in ['', 'nan']:
                            try:
                                link_idx = int(float(current_loan_link))
                                if link_idx < len(loans_df):
                                    loan = loans_df.iloc[link_idx]
                                    st.info(f"🏦 Linked to: {loan['Description']}")
                            except:
                                pass

            # Export options
            st.markdown("---")
            col_export1, col_export2, col_export3 = st.columns(3)

            with col_export1:
                csv_export = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Export to CSV",
                    data=csv_export,
                    file_name=f"transactions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col_export2:
                if st.button("🔄 Refresh Data", use_container_width=True):
                    st.rerun()

            with col_export3:
                if st.button("🗑️ Clear All Filters", use_container_width=True):
                    st.rerun()
        else:
            st.info(
                "📊 No transactions yet. Import or add transactions to get started.")

    with tab4:
        st.subheader("🏷️ Tag Management")

        if not transactions_df.empty:
            # Extract all unique tags
            all_tags = set()
            for tags_str in transactions_df['Tags'].dropna():
                if tags_str:
                    tags = [t.strip() for t in str(tags_str).split(',')]
                    all_tags.update(tags)

            all_tags = sorted([t for t in all_tags if t]
                              )  # Remove empty strings

            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(f"### All Tags ({len(all_tags)})")

                if all_tags:
                    # Display tags with transaction count
                    tag_stats = []
                    for tag in all_tags:
                        count = transactions_df['Tags'].str.contains(
                            tag, case=False, na=False).sum()
                        tag_stats.append({'Tag': tag, 'Count': count})

                    tag_df = pd.DataFrame(tag_stats).sort_values(
                        'Count', ascending=False)

                    st.dataframe(
                        tag_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            'Tag': st.column_config.TextColumn('Tag', width='large'),
                            'Count': st.column_config.NumberColumn('Transactions', width='small')
                        }
                    )

                    st.markdown("---")

                    # Bulk tag operations
                    st.markdown("### Bulk Tag Operations")

                    col_a, col_b = st.columns(2)

                    with col_a:
                        old_tag = st.selectbox("Replace tag", [""] + all_tags)
                        new_tag = st.text_input("With new tag")

                        if st.button("🔄 Replace All", use_container_width=True):
                            if old_tag and new_tag:
                                for idx, row in transactions_df.iterrows():
                                    tags_str = str(row.get('Tags', ''))
                                    if old_tag in tags_str:
                                        new_tags_str = tags_str.replace(
                                            old_tag, new_tag)
                                        transactions_df.at[idx,
                                                           'Tags'] = new_tags_str
                                save_transactions(transactions_df)
                                st.success(
                                    f"Replaced '{old_tag}' with '{new_tag}'")
                                st.rerun()

                    with col_b:
                        remove_tag = st.selectbox(
                            "Remove tag", [""] + all_tags, key="remove_tag")

                        if st.button("🗑️ Remove All", use_container_width=True):
                            if remove_tag:
                                for idx, row in transactions_df.iterrows():
                                    tags_str = str(row.get('Tags', ''))
                                    if remove_tag in tags_str:
                                        tags_list = [t.strip()
                                                     for t in tags_str.split(',')]
                                        tags_list = [
                                            t for t in tags_list if t != remove_tag]
                                        transactions_df.at[idx, 'Tags'] = ','.join(
                                            tags_list)
                                save_transactions(transactions_df)
                                st.success(f"Removed tag '{remove_tag}'")
                                st.rerun()

                    st.markdown("---")

                    # Filter by tag
                    st.markdown("### View Transactions by Tag")
                    selected_tag = st.selectbox("Select tag to view", [
                                                ""] + all_tags, key="view_tag")

                    if selected_tag:
                        tagged_transactions = transactions_df[
                            transactions_df['Tags'].str.contains(
                                selected_tag, case=False, na=False)
                        ].sort_values('Date', ascending=False)

                        st.markdown(
                            f"**{len(tagged_transactions)} transactions with tag '{selected_tag}'**")

                        for idx, row in tagged_transactions.iterrows():
                            amount_color = "#10b981" if row['Type'] == 'income' else "#ef4444"
                            amount_sign = "+" if row['Type'] == 'income' else "-"

                            st.markdown(f"""
                            <div style='background: var(--bg-tertiary); padding: 12px; border-radius: 8px; margin: 8px 0;'>
                                <div style='display: flex; justify-content: space-between;'>
                                    <div>
                                        <strong>{row['Description']}</strong><br>
                                        <span style='font-size: 11px; color: var(--text-secondary);'>
                                            {row['Date'].strftime('%b %d, %Y')} • {row['Category']}
                                        </span>
                                    </div>
                                    <div style='text-align: right;'>
                                        <strong style='color: {amount_color};'>{amount_sign}${abs(row['Amount']):,.2f}</strong>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    st.info(
                        "No tags found. Add tags to transactions to organize them better!")

            with col2:
                st.markdown("### 💡 Tag Suggestions")

                suggested_tags = [
                    "🍽️ dining-out",
                    "🏠 home",
                    "💼 work",
                    "✈️ vacation",
                    "🎁 gift",
                    "🚗 car",
                    "💊 medical",
                    "📚 education",
                    "💪 fitness",
                    "🎮 hobby",
                    "🔧 maintenance",
                    "💳 subscription",
                    "🎉 celebration",
                    "🏥 emergency",
                    "📱 tech"
                ]

                st.markdown("""
                <div class='insight-card'>
                    <strong>Popular Tags:</strong><br>
                """, unsafe_allow_html=True)

                for tag in suggested_tags[:10]:
                    st.markdown(f"• {tag}")

                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("---")

                st.markdown("### 🎯 Tag Tips")
                st.markdown("""
                <div class='insight-card' style='font-size: 12px;'>
                    • Use tags for flexible categorization<br>
                    • Tags work across categories<br>
                    • Use comma-separated format<br>
                    • Keep tags short and consistent<br>
                    • Examples: work, personal, urgent, tax-deductible
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Add transactions first to manage tags!")

    with tab5:
        st.subheader("🔗 Linked Payment Analysis")
        
        # Recurring payments analysis
        st.markdown("### 🔄 Recurring Payments Tracking")

        
        
        if not recurring_df.empty:
            for rec_idx, rec in recurring_df.iterrows():
                linked_transactions = transactions_df[transactions_df['Linked_Recurring_ID'] == str(rec_idx)]
                
                if not linked_transactions.empty:
                    with st.expander(f"{rec['Description']} - {len(linked_transactions)} payments tracked"):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            total_paid = abs(linked_transactions['Amount'].sum())
                            avg_payment = abs(linked_transactions['Amount'].mean())
                            
                            st.metric("Total Paid", f"${total_paid:,.2f}")
                            st.metric("Average Payment", f"${avg_payment:,.2f}")
                            st.metric("Expected", f"${abs(rec['Amount']):,.2f}")
                            
                            if abs(avg_payment - abs(rec['Amount'])) > 0.01:
                                variance = ((avg_payment - abs(rec['Amount'])) / abs(rec['Amount']) * 100)
                                st.metric("Variance", f"{variance:+.1f}%")
                        
                        with col_b:
                            # Payment timeline
                            payment_dates = linked_transactions.sort_values('Date')
                            
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=payment_dates['Date'],
                                y=payment_dates['Amount'].abs(),
                                mode='lines+markers',
                                name='Payments',
                                line=dict(color='#6366f1', width=2),
                                marker=dict(size=8)
                            ))
                            
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'),
                                height=250,
                                yaxis_title="Amount ($)",
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # List payments
                        st.markdown("**Payment History:**")
                        for _, trans in linked_transactions.sort_values('Date', ascending=False).iterrows():
                            st.markdown(f"• {trans['Date'].strftime('%b %d, %Y')} - ${abs(trans['Amount']):,.2f}")
        
        st.markdown("---")
        
        # Loan payments analysis
        st.markdown("### 🏦 Loan Payments Tracking")
        
        if not loans_df.empty:
            for loan_idx, loan in loans_df.iterrows():
                # linked_transactions = filtered_transactions[
                #     filtered_transactions['Linked_Loan_ID'] == str(loan_idx)
                # ]
                linked_transactions = transactions_df[transactions_df['Linked_Recurring_ID'] == str(rec_idx)]
                
                if not linked_transactions.empty:
                    with st.expander(f"{loan['Description']} - {len(linked_transactions)} payments made"):
                        col_a, col_b = st.columns(2)
                        
                        with col_a:
                            total_paid = abs(linked_transactions['Amount'].sum())
                            principal_paid = total_paid  # Simplified
                            remaining = loan.get('Remaining', loan['Principal'])
                            
                            st.metric("Total Paid", f"${total_paid:,.2f}")
                            st.metric("Principal Remaining", f"${remaining:,.2f}")
                            
                            progress = ((loan['Principal'] - remaining) / loan['Principal'] * 100) if loan['Principal'] > 0 else 0
                            st.progress(progress / 100)
                            st.caption(f"{progress:.1f}% paid off")
                        
                        with col_b:
                            # Payment consistency
                            payment_dates = linked_transactions.sort_values('Date')
                            
                            fig = go.Figure()
                            fig.add_trace(go.Bar(
                                x=payment_dates['Date'],
                                y=payment_dates['Amount'].abs(),
                                name='Payments',
                                marker_color='#10b981'
                            ))
                            
                            fig.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                font=dict(color='white'),
                                height=250,
                                yaxis_title="Amount ($)",
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Payment schedule
                        st.markdown("**Payment History:**")
                        for _, trans in linked_transactions.sort_values('Date', ascending=False).iterrows():
                            st.markdown(f"• {trans['Date'].strftime('%b %d, %Y')} - ${abs(trans['Amount']):,.2f}")

# BUDGETS PAGE
elif page == "💳 Budgets":
    st.title("Budget Management")
    st.markdown("*Plan and track your spending*")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📊 Monthly Budget Overview")

        # Calculate current month spending
        current_month_start = datetime.now().replace(day=1)
        current_spending = transactions_df[
            (transactions_df['Date'] >= current_month_start) &
            (transactions_df['Type'] == 'expense')
        ].groupby('Category')['Amount'].sum().abs()

        total_budget = budgets_df['Budget'].sum()
        total_spent = current_spending.sum()
        remaining = total_budget - total_spent

        # Overall progress
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Total Budget", f"${total_budget:,.0f}")
        with col_b:
            st.metric("Spent", f"${total_spent:,.0f}")
        with col_c:
            st.metric("Remaining", f"${remaining:,.0f}")
        with col_d:
            pct_left = (remaining / total_budget *
                        100) if total_budget > 0 else 0
            st.metric("% Left", f"{pct_left:.0f}%")

        # Overall progress bar
        overall_pct = (total_spent / total_budget *
                       100) if total_budget > 0 else 0
        st.progress(min(overall_pct / 100, 1.0))
        st.caption(f"{overall_pct:.1f}% of total budget used this month")

        st.markdown("---")

        # Category breakdown with cards
        st.subheader("📋 Category Breakdown")

        for _, budget_row in budgets_df.iterrows():
            category = budget_row['Category']
            budget = budget_row['Budget']
            spent = current_spending.get(category, 0)
            remaining_cat = budget - spent
            pct = (spent / budget * 100) if budget > 0 else 0

            # Color coding
            if pct < 70:
                color = "#10b981"
                status = "✅"
            elif pct < 90:
                color = "#f59e0b"
                status = "⚠️"
            else:
                color = "#ef4444"
                status = "🚨"

            # FIX: Convert numpy bool to Python bool
            with st.expander(f"{status} {category} - {pct:.0f}% used", expanded=bool(pct >= 90)):
                col_i, col_ii = st.columns([3, 1])

                with col_i:
                    st.markdown(f"""
                    <div style='margin-bottom: 12px;'>
                        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                            <span style='font-weight: 600;'>Spent: ${spent:,.2f}</span>
                            <span style='font-weight: 600;'>Budget: ${budget:,.2f}</span>
                        </div>
                        <div style='background: var(--bg-secondary); height: 12px; border-radius: 6px; overflow: hidden;'>
                            <div style='background: {color}; height: 100%; width: {min(pct, 100):.1f}%; transition: width 0.3s ease;'></div>
                        </div>
                        <div style='margin-top: 8px; display: flex; justify-content: space-between;'>
                            <span style='font-size: 12px; color: {color};'>{pct:.1f}% used</span>
                            <span style='font-size: 12px; color: var(--text-secondary);'>${remaining_cat:,.2f} remaining</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Recent transactions in category
                    cat_transactions = transactions_df[
                        (transactions_df['Date'] >= current_month_start) &
                        (transactions_df['Category'] == category) &
                        (transactions_df['Type'] == 'expense')
                    ].sort_values('Date', ascending=False)

                    if not cat_transactions.empty:
                        st.markdown("**Recent transactions:**")
                        for _, trans in cat_transactions.head(5).iterrows():
                            st.markdown(f"""
                            <div style='padding: 8px; background: var(--bg-tertiary); border-radius: 6px; margin: 4px 0;'>
                                <span style='font-weight: 500;'>{trans['Description']}</span>
                                <span style='float: right; color: {color};'>${abs(trans['Amount']):.2f}</span><br>
                                <span style='font-size: 11px; color: var(--text-secondary);'>
                                    {trans['Date'].strftime('%b %d')} • {trans['Account']}
                                </span>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.info("No transactions in this category yet")

                with col_ii:
                    st.markdown("**Adjust Budget**")
                    new_budget = st.number_input(
                        "New amount",
                        value=float(budget),
                        step=50.0,
                        key=f"budget_{category}"
                    )
                    if st.button("💾 Update", key=f"update_{category}", use_container_width=True):
                        budgets_df.loc[budgets_df['Category']
                                       == category, 'Budget'] = new_budget
                        save_budgets(budgets_df)
                        st.success("Updated!")
                        st.rerun()

                    # Alert threshold
                    threshold = budget_row.get('Alert_Threshold', 90)
                    new_threshold = st.slider(
                        "Alert at %",
                        min_value=50,
                        max_value=100,
                        value=int(threshold),
                        key=f"threshold_{category}"
                    )
                    if new_threshold != threshold:
                        budgets_df.loc[budgets_df['Category'] ==
                                       category, 'Alert_Threshold'] = new_threshold
                        save_budgets(budgets_df)
                        st.rerun()


# GOALS PAGE
elif page == "🎯 Goals":
    st.title("Financial Goals")
    st.markdown("*Track your financial milestones*")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("🎯 Your Goals")

        if not goals_df.empty:
            # Sort by priority
            goals_sorted = goals_df.sort_values('Priority', ascending=True)

            for idx, goal in goals_sorted.iterrows():
                progress = (goal['Current_Amount'] / goal['Target_Amount'] * 100) if goal['Target_Amount'] > 0 else 0
                days_left = (goal['Target_Date'] - datetime.now()).days
                remaining = goal['Target_Amount'] - goal['Current_Amount']

                # Calculate monthly savings needed
                months_left = max(1, days_left / 30)
                monthly_needed = remaining / months_left if months_left > 0 else remaining

                # Priority color
                if goal['Priority'] == 'High':
                    priority_color = "#ef4444"
                    priority_icon = "🔴"
                elif goal['Priority'] == 'Medium':
                    priority_color = "#f59e0b"
                    priority_icon = "🟡"
                else:
                    priority_color = "#10b981"
                    priority_icon = "🟢"

                # Status
                if progress >= 100:
                    status = "✅ Complete"
                    status_color = "#10b981"
                elif days_left < 0:
                    status = "⏰ Overdue"
                    status_color = "#ef4444"
                elif days_left < 30:
                    status = "⚠️ Due Soon"
                    status_color = "#f59e0b"
                else:
                    status = "📈 In Progress"
                    status_color = "#6366f1"

                with st.expander(f"{priority_icon} {goal['Goal_Name']} - {progress:.0f}%", expanded=bool(progress < 100)):
                    # Goal header
                    st.markdown(f"""
                    <div style='background: var(--bg-tertiary); padding: 16px; border-radius: 12px; margin-bottom: 16px;'>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;'>
                            <div>
                                <h3 style='margin: 0;'>{goal['Goal_Name']}</h3>
                                <span style='color: var(--text-secondary); font-size: 13px;'>{goal['Category']}</span>
                            </div>
                            <div style='text-align: right;'>
                                <span style='color: {status_color}; font-weight: 600;'>{status}</span><br>
                                <span style='color: {priority_color}; font-size: 12px;'>{goal['Priority']} Priority</span>
                            </div>
                        </div>
                        <div style='margin: 16px 0;'>
                            <div style='background: var(--bg-secondary); height: 12px; border-radius: 6px; overflow: hidden;'>
                                <div style='background: linear-gradient(90deg, #6366f1, #8b5cf6);
                                            height: 100%; width: {min(progress, 100):.1f}%; transition: width 0.3s ease;'></div>
                            </div>
                        </div>
                        <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 16px;'>
                            <div>
                                <span style='color: var(--text-secondary); font-size: 12px;'>Current</span><br>
                                <strong style='font-size: 18px;'>${goal['Current_Amount']:,.2f}</strong>
                            </div>
                            <div>
                                <span style='color: var(--text-secondary); font-size: 12px;'>Target</span><br>
                                <strong style='font-size: 18px;'>${goal['Target_Amount']:,.2f}</strong>
                            </div>
                            <div>
                                <span style='color: var(--text-secondary); font-size: 12px;'>Remaining</span><br>
                                <strong style='font-size: 18px; color: #6366f1;'>${remaining:,.2f}</strong>
                            </div>
                        </div>
                        <div style='margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border);'>
                            <div style='display: flex; justify-content: space-between;'>
                                <div>
                                    <span style='color: var(--text-secondary); font-size: 12px;'>Target Date</span><br>
                                    <strong>{goal['Target_Date'].strftime('%B %d, %Y')}</strong>
                                </div>
                                <div style='text-align: right;'>
                                    <span style='color: var(--text-secondary); font-size: 12px;'>Days Left</span><br>
                                    <strong style='color: {"#ef4444" if days_left < 0 else "#f59e0b" if days_left < 30 else "#10b981"};'>
                                        {abs(days_left)} days {"overdue" if days_left < 0 else ""}
                                    </strong>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Action buttons
                    col_a, col_b, col_c = st.columns(3)

                    with col_a:
                        contribution = st.number_input("Add Amount ($)", min_value=0.0, step=10.0, key=f"contrib_{idx}")
                        if st.button("💰 Contribute", key=f"add_{idx}", use_container_width=True):
                            goals_df.at[idx, 'Current_Amount'] += contribution
                            save_goals(goals_df)
                            st.success(f"Added ${contribution:,.2f}!")
                            st.rerun()

                    with col_b:
                        new_target_date = st.date_input("Extend Date", value=goal['Target_Date'].date(), key=f"date_{idx}")
                        if st.button("📅 Update Date", key=f"update_date_{idx}", use_container_width=True):
                            goals_df.at[idx, 'Target_Date'] = pd.to_datetime(new_target_date)
                            save_goals(goals_df)
                            st.success("Date updated!")
                            st.rerun()

                    with col_c:
                        if st.button("🗑️ Delete Goal", key=f"del_goal_{idx}", use_container_width=True):
                            goals_df = goals_df.drop(idx).reset_index(drop=True)
                            save_goals(goals_df)
                            st.rerun()

                    # Insights
                    if progress < 100 and days_left > 0:
                        st.markdown(f"""
                        <div class='insight-card' style='margin-top: 16px;'>
                            <strong>💡 To reach your goal:</strong><br>
                            <span style='font-size: 13px;'>
                            Save <strong>${monthly_needed:,.2f}/month</strong>
                            (${monthly_needed/30:.2f}/day) for the next {months_left:.0f} months
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("🎯 No goals yet. Create your first financial goal!")

    with col2:
        st.subheader("➕ Create New Goal")

        with st.form("goal_form", clear_on_submit=True):
            goal_name = st.text_input("Goal Name", placeholder="e.g., Emergency Fund")

            col_a, col_b = st.columns(2)
            with col_a:
                target_amount = st.number_input("Target Amount ($)", min_value=1.0, step=100.0, value=1000.0)
            with col_b:
                current_amount = st.number_input("Current Amount ($)", min_value=0.0, step=100.0, value=0.0)

            target_date = st.date_input("Target Date", value=datetime.now() + timedelta(days=365))

            col_c, col_d = st.columns(2)
            with col_c:
                priority = st.selectbox("Priority", ["High", "Medium", "Low"])
            with col_d:
                category = st.selectbox("Category", ["Emergency Fund", "Retirement", "Home", "Education", "Vacation", "Investment", "Other"])

            if st.form_submit_button("🎯 Create Goal", type="primary", use_container_width=True):
                if goal_name and target_amount > 0:
                    new_goal = pd.DataFrame([{
                        'Goal_Name': goal_name,
                        'Target_Amount': target_amount,
                        'Current_Amount': current_amount,
                        'Target_Date': pd.to_datetime(target_date),
                        'Priority': priority,
                        'Category': category
                    }])

                    if goals_df.empty:
                        goals_df = new_goal
                    else:
                        goals_df = pd.concat([goals_df, new_goal], ignore_index=True)

                    save_goals(goals_df)
                    st.success("✅ Goal created!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("Please fill in all required fields")

        st.markdown("---")

        # Goal templates
        st.subheader("📋 Quick Templates")

        templates = {
            "🚨 Emergency Fund": {"amount": 10000, "months": 12, "category": "Emergency Fund"},
            "🏠 House Down Payment": {"amount": 50000, "months": 36, "category": "Home"},
            "🎓 Education Fund": {"amount": 20000, "months": 24, "category": "Education"},
            "✈️ Vacation": {"amount": 5000, "months": 12, "category": "Vacation"},
            "📈 Investment": {"amount": 15000, "months": 18, "category": "Investment"}
        }

        for template_name, template_data in templates.items():
            if st.button(template_name, use_container_width=True):
                new_goal = pd.DataFrame([{
                    'Goal_Name': template_name.split(' ', 1)[1],
                    'Target_Amount': template_data['amount'],
                    'Current_Amount': 0,
                    'Target_Date': pd.to_datetime(datetime.now() + timedelta(days=template_data['months']*30)),
                    'Priority': 'Medium',
                    'Category': template_data['category']
                }])

                if goals_df.empty:
                    goals_df = new_goal
                else:
                    goals_df = pd.concat([goals_df, new_goal], ignore_index=True)

                save_goals(goals_df)
                st.success(f"✅ {template_name} goal created!")
                st.rerun()

        st.markdown("---")

        # Goal tips
        st.markdown("### 💡 Goal Tips")
        st.markdown("""
        <div class='insight-card'>
        • Set SMART goals (Specific, Measurable, Achievable, Relevant, Time-bound)<br>
        • Break large goals into smaller milestones<br>
        • Automate savings when possible<br>
        • Review and adjust regularly<br>
        • Celebrate achievements!
        </div>
        """, unsafe_allow_html=True)

# LOANS PAGE
elif page == "🏦 Loans":
    st.title("Loan Management")
    st.markdown("*Track and manage your debts*")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💳 Your Loans")

        if not loans_df.empty:
            total_debt = loans_df['Remaining'].sum(
            ) if 'Remaining' in loans_df.columns else loans_df['Principal'].sum()
            total_monthly = loans_df['Monthly_Payment'].sum()

            # Summary metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Debt", f"${total_debt:,.2f}")
            with col_b:
                st.metric("Monthly Payments", f"${total_monthly:,.2f}")
            with col_c:
                avg_rate = loans_df['Rate'].mean()
                st.metric("Avg Interest Rate", f"{avg_rate:.2f}%")

            st.markdown("---")

            # Display each loan
            for idx, loan in loans_df.iterrows():
                months_elapsed = max(
                    0, (datetime.now() - loan['Start_Date']).days // 30)
                total_months = loan['Term_Years'] * 12
                months_remaining = max(0, total_months - months_elapsed)

                # Calculate remaining balance
                remaining = loan.get('Remaining', loan['Principal'])

                progress_pct = (months_elapsed / total_months *
                                100) if total_months > 0 else 0
                paid_pct = ((loan['Principal'] - remaining) /
                            loan['Principal'] * 100) if loan['Principal'] > 0 else 0

                # Total interest paid
                total_interest = (loan['Monthly_Payment']
                                  * total_months) - loan['Principal']
                interest_paid = loan['Monthly_Payment'] * \
                    months_elapsed - (loan['Principal'] - remaining)

                # Loan type color
                loan_type = loan.get('Type', 'Personal')
                if loan_type == 'Mortgage':
                    type_color = "#6366f1"
                    type_icon = "🏠"
                elif loan_type == 'Auto':
                    type_color = "#10b981"
                    type_icon = "🚗"
                elif loan_type == 'Student':
                    type_color = "#f59e0b"
                    type_icon = "🎓"
                else:
                    type_color = "#ef4444"
                    type_icon = "💳"

                with st.expander(f"{type_icon} {loan['Description']} - ${remaining:,.2f} remaining", expanded=True):
                    st.markdown(f"""
                    <div style='background: var(--bg-tertiary); padding: 16px; border-radius: 12px; margin-bottom: 16px;'>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;'>
                            <div>
                                <h3 style='margin: 0;'>{loan['Description']}</h3>
                                <span style='color: {type_color}; font-size: 13px;'>{loan_type} Loan</span>
                            </div>
                            <div style='text-align: right;'>
                                <h2 style='margin: 0; color: #ef4444;'>${remaining:,.2f}</h2>
                                <span style='color: var(--text-secondary); font-size: 12px;'>Remaining</span>
                            </div>
                        </div>
                        
                        <div style='margin: 16px 0;'>
                            <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                                <span style='font-size: 13px;'>Payoff Progress</span>
                                <span style='font-size: 13px; font-weight: 600;'>{paid_pct:.1f}% paid</span>
                            </div>
                            <div style='background: var(--bg-secondary); height: 10px; border-radius: 5px; overflow: hidden;'>
                                <div style='background: linear-gradient(90deg, #10b981, #059669); 
                                            height: 100%; width: {paid_pct:.1f}%; transition: width 0.3s ease;'></div>
                            </div>
                        </div>
                        
                        <div style='display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 16px;'>
                            <div>
                                <span style='color: var(--text-secondary); font-size: 11px;'>Original</span><br>
                                <strong style='font-size: 16px;'>${loan['Principal']:,.0f}</strong>
                            </div>
                            <div>
                                <span style='color: var(--text-secondary); font-size: 11px;'>Rate</span><br>
                                <strong style='font-size: 16px;'>{loan['Rate']:.2f}%</strong>
                            </div>
                            <div>
                                <span style='color: var(--text-secondary); font-size: 11px;'>Monthly</span><br>
                                <strong style='font-size: 16px;'>${loan['Monthly_Payment']:,.0f}</strong>
                            </div>
                            <div>
                                <span style='color: var(--text-secondary); font-size: 11px;'>Months Left</span><br>
                                <strong style='font-size: 16px;'>{months_remaining}</strong>
                            </div>
                        </div>
                        
                        <div style='margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border);'>
                            <div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;'>
                                <div>
                                    <span style='color: var(--text-secondary); font-size: 11px;'>Start Date</span><br>
                                    <strong style='font-size: 13px;'>{loan['Start_Date'].strftime('%b %Y')}</strong>
                                </div>
                                <div>
                                    <span style='color: var(--text-secondary); font-size: 11px;'>Payoff Date</span><br>
                                    <strong style='font-size: 13px;'>
                                        {(loan['Start_Date'] + timedelta(days=total_months*30)).strftime('%b %Y')}
                                    </strong>
                                </div>
                                <div>
                                    <span style='color: var(--text-secondary); font-size: 11px;'>Total Interest</span><br>
                                    <strong style='font-size: 13px; color: #ef4444;'>${total_interest:,.0f}</strong>
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Extra payment calculator
                    st.markdown("### 💡 Extra Payment Impact")

                    col_a, col_b = st.columns(2)

                    with col_a:
                        extra_payment = st.number_input(
                            "Extra Monthly Payment ($)",
                            min_value=0.0,
                            step=50.0,
                            key=f"extra_{idx}"
                        )

                    with col_b:
                        one_time_payment = st.number_input(
                            "One-time Payment ($)",
                            min_value=0.0,
                            step=100.0,
                            key=f"onetime_{idx}"
                        )

                    if extra_payment > 0 or one_time_payment > 0:
                        # Simplified calculation
                        new_monthly = loan['Monthly_Payment'] + extra_payment
                        adjusted_remaining = remaining - one_time_payment

                        if adjusted_remaining > 0 and new_monthly > 0:
                            # Estimate new payoff time
                            monthly_rate = loan['Rate'] / 12 / 100
                            if monthly_rate > 0:
                                new_months = - \
                                    np.log(
                                        1 - (adjusted_remaining * monthly_rate / new_monthly)) / np.log(1 + monthly_rate)
                            else:
                                new_months = adjusted_remaining / new_monthly

                            months_saved = months_remaining - new_months
                            interest_saved = (loan['Monthly_Payment'] * months_remaining) - (
                                new_monthly * new_months) - one_time_payment

                            st.markdown(f"""
                            <div class='success-card' style='margin-top: 12px;'>
                                <strong>💰 Savings Impact</strong><br>
                                <span style='font-size: 13px;'>
                                • Pay off <strong>{months_saved:.0f} months earlier</strong><br>
                                • Save <strong>${max(0, interest_saved):,.2f}</strong> in interest<br>
                                • New payoff date: <strong>{(datetime.now() + timedelta(days=new_months*30)).strftime('%B %Y')}</strong>
                                </span>
                            </div>
                            """, unsafe_allow_html=True)

                    # Action buttons
                    col_i, col_ii, col_iii = st.columns(3)

                    with col_i:
                        payment_made = st.number_input(
                            "Record Payment ($)", min_value=0.0, step=loan['Monthly_Payment'], key=f"payment_{idx}")
                        if st.button("💳 Record", key=f"record_{idx}", use_container_width=True):
                            if payment_made > 0:
                                loans_df.at[idx, 'Remaining'] = max(
                                    0, remaining - payment_made)
                                save_loans(loans_df)
                                st.success(
                                    f"Payment of ${payment_made:,.2f} recorded!")
                                st.rerun()

                    with col_ii:
                        if st.button("📊 Amortization", key=f"amort_{idx}", use_container_width=True):
                            st.info("Amortization schedule feature coming soon!")

                    with col_iii:
                        if st.button("🗑️ Delete Loan", key=f"del_loan_{idx}", use_container_width=True):
                            loans_df = loans_df.drop(
                                idx).reset_index(drop=True)
                            save_loans(loans_df)
                            st.rerun()

            # Debt payoff visualization
            st.markdown("---")
            st.subheader("📊 Debt Payoff Timeline")

            # Create payoff chart
            payoff_data = []
            for _, loan in loans_df.iterrows():
                months_left = max(
                    0, (loan['Term_Years'] * 12) - ((datetime.now() - loan['Start_Date']).days // 30))
                payoff_date = datetime.now() + timedelta(days=months_left * 30)
                payoff_data.append({
                    'Loan': loan['Description'],
                    'Payoff Date': payoff_date,
                    'Amount': loan.get('Remaining', loan['Principal'])
                })

            if payoff_data:
                payoff_df = pd.DataFrame(payoff_data)
                fig = px.bar(
                    payoff_df,
                    x='Payoff Date',
                    y='Amount',
                    color='Loan',
                    title="Projected Debt Payoff",
                    labels={'Amount': 'Remaining Balance ($)'}
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white'),
                    height=350
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🏦 No loans tracked. Add a loan to start managing debt.")

    with col2:
        st.subheader("➕ Add New Loan")

        with st.form("loan_form", clear_on_submit=True):
            loan_description = st.text_input(
                "Loan Name", placeholder="e.g., Car Loan")

            loan_type = st.selectbox(
                "Loan Type", ["Personal", "Mortgage", "Auto", "Student", "Credit Card", "Other"])

            col_a, col_b = st.columns(2)
            with col_a:
                principal = st.number_input(
                    "Principal Amount ($)", min_value=0.0, step=1000.0, value=10000.0)
            with col_b:
                rate = st.number_input(
                    "Interest Rate (%)", min_value=0.0, step=0.1, value=5.0)

            col_c, col_d = st.columns(2)
            with col_c:
                term_years = st.number_input(
                    "Term (Years)", min_value=1, step=1, value=5)
            with col_d:
                start_date = st.date_input("Start Date", value=datetime.now())

            if st.form_submit_button("➕ Add Loan", type="primary", use_container_width=True):
                if loan_description and principal > 0:
                    monthly_payment = calculate_monthly_loan_payment(
                        principal, rate, term_years)

                    new_loan = pd.DataFrame([{
                        'Description': loan_description,
                        'Principal': principal,
                        'Rate': rate,
                        'Term_Years': term_years,
                        'Start_Date': pd.to_datetime(start_date),
                        'Monthly_Payment': monthly_payment,
                        'Remaining': principal,
                        'Type': loan_type
                    }])

                    if loans_df.empty:
                        loans_df = new_loan
                    else:
                        loans_df = pd.concat(
                            [loans_df, new_loan], ignore_index=True)

                    save_loans(loans_df)
                    st.success(
                        f"✅ Loan added! Monthly payment: ${monthly_payment:,.2f}")
                    st.rerun()
                else:
                    st.error("Please fill in all required fields")

        st.markdown("---")

        # Debt payoff strategies
        st.subheader("💡 Payoff Strategies")

        st.markdown("""
        <div class='insight-card'>
            <strong>🔥 Debt Avalanche</strong><br>
            <span style='font-size: 12px;'>Pay off highest interest rate first to minimize total interest</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='insight-card'>
            <strong>⛄ Debt Snowball</strong><br>
            <span style='font-size: 12px;'>Pay off smallest balance first for psychological wins</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Loan calculator
        st.subheader("🧮 Loan Calculator")

        calc_amount = st.number_input(
            "Loan Amount", min_value=1000.0, step=1000.0, value=20000.0, key="calc_amount")
        calc_rate = st.number_input(
            "Interest Rate (%)", min_value=0.0, step=0.1, value=5.5, key="calc_rate")
        calc_years = st.number_input(
            "Term (Years)", min_value=1, step=1, value=5, key="calc_years")

        calc_payment = calculate_monthly_loan_payment(
            calc_amount, calc_rate, calc_years)
        total_paid = calc_payment * calc_years * 12
        total_interest = total_paid - calc_amount

        st.markdown(f"""
        <div class='glass-card' style='padding: 16px; margin-top: 12px;'>
            <strong>Monthly Payment</strong><br>
            <h2 style='margin: 8px 0; color: #6366f1;'>${calc_payment:,.2f}</h2>
            <div style='margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border);'>
                <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                    <span style='color: var(--text-secondary);'>Total Paid</span>
                    <strong>${total_paid:,.2f}</strong>
                </div>
                <div style='display: flex; justify-content: space-between; margin: 8px 0;'>
                    <span style='color: var(--text-secondary);'>Total Interest</span>
                    <strong style='color: #ef4444;'>${total_interest:,.2f}</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# RECURRING TRANSACTIONS PAGE
elif page == "🔄 Recurring":
    st.title("Recurring Transactions")
    st.markdown("*Manage subscriptions and recurring payments*")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📅 Your Recurring Transactions")

        if not recurring_df.empty:
            # Calculate monthly recurring cost
            monthly_cost = 0
            for _, rec in recurring_df[recurring_df['Active'] == True].iterrows():
                if rec['Frequency'] == 'Monthly':
                    monthly_cost += abs(rec['Amount'])
                elif rec['Frequency'] == 'Weekly':
                    monthly_cost += abs(rec['Amount']) * 4.33
                elif rec['Frequency'] == 'Yearly':
                    monthly_cost += abs(rec['Amount']) / 12
                elif rec['Frequency'] == 'Quarterly':
                    monthly_cost += abs(rec['Amount']) / 3

            yearly_cost = monthly_cost * 12

            # Summary
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                active_count = recurring_df[recurring_df['Active']
                                            == True].shape[0]
                st.metric("Active Recurring", active_count)
            with col_b:
                st.metric("Monthly Cost", f"${monthly_cost:,.2f}")
            with col_c:
                st.metric("Yearly Cost", f"${yearly_cost:,.2f}")

            st.markdown("---")

            # Group by frequency
            for frequency in ['Monthly', 'Bi-Weekly','Weekly', 'Yearly', 'Quarterly']:
                freq_items = recurring_df[
                    (recurring_df['Frequency'] == frequency) &
                    (recurring_df['Active'] == True)
                ]

                if not freq_items.empty:
                    st.markdown(f"### {frequency} Transactions")

                    for idx, rec in freq_items.iterrows():
                        days_until = (rec['Next_Date'] - datetime.now()).days

                        if days_until < 0:
                            status_color = "#ef4444"
                            status = "⏰ Overdue"
                        elif days_until < 7:
                            status_color = "#f59e0b"
                            status = "⚠️ Due Soon"
                        else:
                            status_color = "#10b981"
                            status = "✅ Scheduled"

                        # Category icon
                        category_icons = {
                            'Bills & Utilities': '💡',
                            'Entertainment': '🎬',
                            'Shopping': '🛍️',
                            'Healthcare': '⚕️',
                            'Education': '📚',
                            'Transportation': '🚗'
                        }
                        icon = category_icons.get(rec['Category'], '💳')

                        with st.expander(f"{icon} {rec['Description']} - ${abs(rec['Amount']):,.2f}/{frequency}", expanded=(days_until < 7)):
                            col_i, col_ii = st.columns([3, 1])

                            with col_i:
                                st.markdown(f"""
                                <div style='background: var(--bg-tertiary); padding: 16px; border-radius: 12px;'>
                                    <div style='display: flex; justify-content: space-between; margin-bottom: 12px;'>
                                        <div>
                                            <h4 style='margin: 0;'>{rec['Description']}</h4>
                                            <span style='color: var(--text-secondary); font-size: 12px;'>
                                                {rec['Category']} • {rec['Account']}
                                            </span>
                                        </div>
                                        <div style='text-align: right;'>
                                            <h3 style='margin: 0; color: #ef4444;'>${abs(rec['Amount']):,.2f}</h3>
                                            <span style='color: var(--text-secondary); font-size: 11px;'>{frequency}</span>
                                        </div>
                                    </div>
                                    <div style='margin-top: 12px; padding-top: 12px; border-top: 1px solid var(--border);'>
                                        <div style='display: flex; justify-content: space-between;'>
                                            <div>
                                                <span style='color: var(--text-secondary); font-size: 11px;'>Next Payment</span><br>
                                                <strong>{rec['Next_Date'].strftime('%B %d, %Y')}</strong>
                                            </div>
                                            <div style='text-align: right;'>
                                                <span style='color: var(--text-secondary); font-size: 11px;'>Status</span><br>
                                                <strong style='color: {status_color};'>{status}</strong>
                                            </div>
                                        </div>
                                    </div>
                                    <div style='margin-top: 12px;'>
                                        <span style='color: var(--text-secondary); font-size: 11px;'>
                                            {abs(days_until)} days {"overdue" if days_until < 0 else "until next payment"}
                                        </span>
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                            with col_ii:
                                st.markdown("**Actions**")

                                if st.button("✅ Mark Paid", key=f"paid_{idx}", use_container_width=True):
                                    # Update next date based on frequency
                                    if rec['Frequency'] == 'Monthly':
                                        next_date = rec['Next_Date'] + \
                                            timedelta(days=30)
                                    if rec['Frequency'] == 'Bi-Weekly':
                                        next_date = rec['Next_Date'] + \
                                            timedelta(days=14)
                                    elif rec['Frequency'] == 'Weekly':
                                        next_date = rec['Next_Date'] + \
                                            timedelta(days=7)
                                    elif rec['Frequency'] == 'Yearly':
                                        next_date = rec['Next_Date'] + \
                                            timedelta(days=365)
                                    elif rec['Frequency'] == 'Quarterly':
                                        next_date = rec['Next_Date'] + \
                                            timedelta(days=90)

                                    recurring_df.at[idx,
                                                    'Next_Date'] = next_date
                                    save_recurring(recurring_df)

                                    # Add to transactions
                                    new_trans = pd.DataFrame([{
                                        'Date': pd.to_datetime(datetime.now()),
                                        'Description': rec['Description'],
                                        'Amount': rec['Amount'],
                                        'Category': rec['Category'],
                                        'Type': 'expense',
                                        'Account': rec['Account'],
                                        'Source': 'Recurring',
                                        'Tags': 'subscription,recurring'
                                    }])

                                    if transactions_df.empty:
                                        transactions_df = new_trans
                                    else:
                                        transactions_df = pd.concat(
                                            [transactions_df, new_trans], ignore_index=True)
                                    save_transactions(transactions_df)

                                    st.success("Payment recorded!")
                                    st.rerun()

                                if st.button("⏸️ Pause", key=f"pause_{idx}", use_container_width=True):
                                    recurring_df.at[idx, 'Active'] = False
                                    save_recurring(recurring_df)
                                    st.success("Paused!")
                                    st.rerun()

                                if st.button("🗑️ Delete", key=f"del_rec_{idx}", use_container_width=True):
                                    recurring_df = recurring_df.drop(
                                        idx).reset_index(drop=True)
                                    save_recurring(recurring_df)
                                    st.rerun()

                    st.markdown("---")

            # Inactive recurring
            inactive = recurring_df[recurring_df['Active'] == False]
            if not inactive.empty:
                with st.expander(f"⏸️ Paused Transactions ({len(inactive)})"):
                    for idx, rec in inactive.iterrows():
                        col_i, col_ii = st.columns([4, 1])
                        with col_i:
                            st.markdown(
                                f"**{rec['Description']}** - ${abs(rec['Amount']):,.2f}/{rec['Frequency']}")
                        with col_ii:
                            if st.button("▶️ Resume", key=f"resume_{idx}", use_container_width=True):
                                recurring_df.at[idx, 'Active'] = True
                                save_recurring(recurring_df)
                                st.rerun()
        else:
            st.info(
                "🔄 No recurring transactions. Add subscriptions and bills to track them.")

    with col2:
        st.subheader("➕ Add Recurring Transaction")

        with st.form("recurring_form", clear_on_submit=True):
            description = st.text_input(
                "Description", placeholder="e.g., Netflix Subscription")

            col_a, col_b = st.columns(2)
            with col_a:
                amount = st.number_input(
                    "Amount ($)", min_value=0.01, step=1.0, value=15.99)
            with col_b:
                frequency = st.selectbox(
                    "Frequency", ["Monthly", 'Bi-weekly',"Weekly", "Yearly", "Quarterly"])

            category = st.selectbox(
                "Category", budgets_df['Category'].tolist())
            account = st.selectbox("Account", accounts_df['Account_Name'].tolist(
            ) if not accounts_df.empty else ['Default'])
            next_date = st.date_input(
                "Next Payment Date", value=datetime.now())

            if st.form_submit_button("➕ Add Recurring", type="primary", use_container_width=True):
                if description and amount > 0:
                    new_recurring = pd.DataFrame([{
                        'Description': description,
                        'Amount': -abs(amount),
                        'Category': category,
                        'Frequency': frequency,
                        'Next_Date': pd.to_datetime(next_date),
                        'Account': account,
                        'Active': True
                    }])

                    if recurring_df.empty:
                        recurring_df = new_recurring
                    else:
                        recurring_df = pd.concat(
                            [recurring_df, new_recurring], ignore_index=True)

                    save_recurring(recurring_df)
                    st.success("✅ Recurring transaction added!")
                    st.rerun()
                else:
                    st.error("Please fill in all required fields")

        st.markdown("---")

        # Popular subscriptions
        st.subheader("📋 Popular Subscriptions")

        popular_subs = {
            "Netflix": 15.99,
            "Spotify": 10.99,
            "Amazon Prime": 14.99,
            "Disney+": 7.99,
            "YouTube Premium": 11.99,
            "Apple Music": 10.99,
            "Hulu": 7.99,
            "HBO Max": 15.99
        }

        for sub_name, sub_price in popular_subs.items():
            if st.button(f"{sub_name} - ${sub_price}", use_container_width=True):
                new_recurring = pd.DataFrame([{
                    'Description': sub_name,
                    'Amount': -sub_price,
                    'Category': 'Entertainment',
                    'Frequency': 'Monthly',
                    'Next_Date': pd.to_datetime(datetime.now() + timedelta(days=30)),
                    'Account': accounts_df['Account_Name'].iloc[0] if not accounts_df.empty else 'Default',
                    'Active': True
                }])

                if recurring_df.empty:
                    recurring_df = new_recurring
                else:
                    recurring_df = pd.concat(
                        [recurring_df, new_recurring], ignore_index=True)

                save_recurring(recurring_df)
                st.success(f"✅ {sub_name} added!")
                st.rerun()

        st.markdown("---")

        # Subscription insights
        st.subheader("💡 Subscription Tips")

        if not recurring_df.empty:
            active_recurring = recurring_df[recurring_df['Active'] == True]
            unused_threshold = datetime.now() - timedelta(days=60)

            st.markdown("""
            <div class='insight-card'>
            • Review subscriptions quarterly<br>
            • Cancel unused services<br>
            • Look for annual discounts<br>
            • Share family plans<br>
            • Track free trial end dates
            </div>
            """, unsafe_allow_html=True)

# AI ASSISTANT PAGE
elif page == "🤖 AI Assistant":
    st.title("AI Financial Assistant")
    st.markdown("*Get personalized financial advice powered by Gemini AI*")

    if not AI_ENABLED:
        st.error(
            "⚠️ Gemini API not configured. Please set GEMINI_API_KEY environment variable.")
        st.code("export GEMINI_API_KEY='your_api_key_here'")
        st.markdown(
            "[Get your API key from Google AI Studio](https://makersuite.google.com/app/apikey)")
        st.stop()

    # Prepare financial context
    net_worth = calculate_net_worth(transactions_df, accounts_df, loans_df)
    current_month_start = datetime.now().replace(day=1)

    monthly_income = transactions_df[
        (transactions_df['Date'] >= current_month_start) &
        (transactions_df['Type'] == 'income')
    ]['Amount'].sum()

    monthly_expenses = abs(transactions_df[
        (transactions_df['Date'] >= current_month_start) &
        (transactions_df['Type'] == 'expense')
    ]['Amount'].sum())

    total_debt = loans_df['Remaining'].sum(
    ) if not loans_df.empty and 'Remaining' in loans_df.columns else 0

    context = f"""
    Net Worth: ${net_worth:,.2f}
    Monthly Income: ${monthly_income:,.2f}
    Monthly Expenses: ${monthly_expenses:,.2f}
    Savings Rate: {((monthly_income - monthly_expenses) / monthly_income * 100) if monthly_income > 0 else 0:.1f}%
    Total Debt: ${total_debt:,.2f}
    Number of Accounts: {len(accounts_df)}
    Number of Goals: {len(goals_df)}
    Active Budget Categories: {len(budgets_df)}
    """

    # Two column layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("💬 Chat with Your Financial Advisor")

        # Chat container
        chat_container = st.container()

        with chat_container:
            # Display chat history
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class='chat-message user-message'>
                        <strong>You:</strong><br>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='chat-message ai-message'>
                        <strong>🤖 AI Advisor:</strong><br>
                        {message['content']}
                    </div>
                    """, unsafe_allow_html=True)

        # Chat input
        user_message = st.text_input(
            "Ask anything about your finances...",
            placeholder="e.g., How can I save more money? Should I pay off debt or invest?",
            key="chat_input"
        )

        col_send, col_clear = st.columns([1, 1])

        with col_send:
            if st.button("📤 Send", type="primary", use_container_width=True):
                if user_message:
                    # Add user message to history
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': user_message
                    })

                    # Get AI response
                    with st.spinner("🤔 AI is thinking..."):
                        ai_response = ai_chat(user_message, context)

                    # Add AI response to history
                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': ai_response
                    })

                    st.rerun()

        with col_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        st.markdown("---")

        # Suggested questions
        st.subheader("💡 Suggested Questions")

        suggested_questions = [
            "How can I improve my savings rate?",
            "What's the best way to pay off my debts?",
            "Am I spending too much in any category?",
            "Should I invest or save for emergencies first?",
            "How can I reduce my monthly expenses?",
            "What are some good financial goals for me?",
            "Is my budget allocation healthy?",
            "How much should I save for retirement?"
        ]

        cols = st.columns(2)
        for idx, question in enumerate(suggested_questions):
            with cols[idx % 2]:
                if st.button(f"❓ {question}", key=f"suggest_{idx}", use_container_width=True):
                    st.session_state.chat_history.append({
                        'role': 'user',
                        'content': question
                    })

                    with st.spinner("🤔 AI is thinking..."):
                        ai_response = ai_chat(question, context)

                    st.session_state.chat_history.append({
                        'role': 'assistant',
                        'content': ai_response
                    })

                    st.rerun()

    with col2:
        st.subheader("📊 AI Insights")

        if st.button("✨ Generate Insights", type="primary", use_container_width=True):
            with st.spinner("Analyzing your finances..."):
                insights = ai_financial_insights(context)

                for insight in insights:
                    if insight.strip():
                        st.markdown(f"""
                        <div class='insight-card'>
                            {insight}
                        </div>
                        """, unsafe_allow_html=True)

        st.markdown("---")

        # Quick actions
        st.subheader("⚡ Quick Actions")

        if st.button("📈 Analyze Spending", use_container_width=True):
            if not transactions_df.empty:
                category_spending = transactions_df[
                    transactions_df['Type'] == 'expense'
                ].groupby('Category')['Amount'].sum().abs()

                prompt = f"Analyze this spending breakdown and provide specific recommendations: {category_spending.to_dict()}"

                with st.spinner("Analyzing..."):
                    response = ai_chat(prompt, context)
                    st.markdown(f"""
                    <div class='ai-message'>
                        {response}
                    </div>
                    """, unsafe_allow_html=True)

        if st.button("🎯 Goal Recommendations", use_container_width=True):
            with st.spinner("Generating recommendations..."):
                prompt = f"Based on my financial situation (Net Worth: ${net_worth:,.2f}, Monthly Income: ${monthly_income:,.2f}, Monthly Expenses: ${monthly_expenses:,.2f}), what financial goals should I set?"
                response = ai_chat(prompt, context)
                st.markdown(f"""
                <div class='ai-message'>
                    {response}
                </div>
                """, unsafe_allow_html=True)

        if st.button("💡 Budget Optimization", use_container_width=True):
            with st.spinner("Optimizing budget..."):
                suggestions = ai_suggest_budget(
                    transactions_df, monthly_income)
                if suggestions:
                    st.markdown(f"""
                    <div class='ai-message'>
                        {suggestions}
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # AI Stats
        st.markdown("### 📈 Your Stats")
        st.metric("Net Worth", f"${net_worth:,.2f}")
        st.metric(
            "Savings Rate", f"{((monthly_income - monthly_expenses) / monthly_income * 100) if monthly_income > 0 else 0:.1f}%")
        st.metric("Monthly Expenses", f"${monthly_expenses:,.2f}")

# ANALYTICS PAGE
elif page == "📈 Analytics":
    st.title("Financial Analytics")
    st.markdown("*Deep dive into your financial data*")

    if transactions_df.empty:
        st.info("📊 Add transactions to see analytics")
    else:
        st.markdown("### 📊 Your Analytics")

    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date", value=datetime.now() - timedelta(days=180))
    with col2:
        end_date = st.date_input("End Date", value=datetime.now())

    # Filter data
    filtered_transactions = transactions_df[
        (transactions_df['Date'] >= pd.to_datetime(start_date)) &
        (transactions_df['Date'] <= pd.to_datetime(end_date))
    ]

    # Summary metrics
    st.subheader("📊 Overview")
    col1, col2, col3, col4 = st.columns(4)

    total_income = filtered_transactions[filtered_transactions['Type']
                                         == 'income']['Amount'].sum()
    total_expenses = abs(
        filtered_transactions[filtered_transactions['Type'] == 'expense']['Amount'].sum())
    net_income = total_income - total_expenses
    avg_daily = total_expenses / max(1, (end_date - start_date).days)

    with col1:
        st.metric("Total Income", f"${total_income:,.2f}")
    with col2:
        st.metric("Total Expenses", f"${total_expenses:,.2f}")
    with col3:
        st.metric("Net Income", f"${net_income:,.2f}",
                  delta=f"{net_income:+,.2f}")
    with col4:
        st.metric("Avg Daily Spend", f"${avg_daily:,.2f}")

    st.markdown("---")

    # Tabs for different analytics
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Trends", "🎯 Categories", "💳 Accounts", "📅 Calendar"])

    with tab1:
        st.subheader("Spending Trends Over Time")

        # Daily spending
        daily_spending = filtered_transactions[filtered_transactions['Type'] == 'expense'].copy(
        )
        daily_spending['Day'] = daily_spending['Date'].dt.date
        daily_summary = daily_spending.groupby(
            'Day')['Amount'].sum().abs().reset_index()
        daily_summary['Day'] = pd.to_datetime(daily_summary['Day'])

        # 7-day moving average
        daily_summary['MA7'] = daily_summary['Amount'].rolling(window=7).mean()

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=daily_summary['Day'],
            y=daily_summary['Amount'],
            mode='markers',
            name='Daily Spending',
            marker=dict(size=6, color='#ef4444', opacity=0.6)
        ))

        fig.add_trace(go.Scatter(
            x=daily_summary['Day'],
            y=daily_summary['MA7'],
            mode='lines',
            name='7-Day Average',
            line=dict(color='#6366f1', width=3)
        ))

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True,
                       gridcolor='rgba(255,255,255,0.1)', title="Amount ($)"),
            height=400,
            hovermode='x unified'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Monthly comparison
        st.subheader("Monthly Comparison")

        monthly_data = filtered_transactions.copy()
        monthly_data['Month'] = monthly_data['Date'].dt.to_period(
            'M').astype(str)
        monthly_comparison = monthly_data.groupby(
            ['Month', 'Type'])['Amount'].sum().reset_index()
        monthly_comparison['Amount'] = monthly_comparison.apply(
            lambda x: abs(
                x['Amount']) if x['Type'] == 'expense' else x['Amount'],
            axis=1
        )

        fig = px.bar(
            monthly_comparison,
            x='Month',
            y='Amount',
            color='Type',
            barmode='group',
            title="Income vs Expenses by Month",
            color_discrete_map={'income': '#10b981', 'expense': '#ef4444'}
        )

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Category Analysis")

        col_a, col_b = st.columns(2)

        with col_a:
            # Pie chart
            category_spending = filtered_transactions[
                filtered_transactions['Type'] == 'expense'
            ].groupby('Category')['Amount'].sum().abs()

            fig = px.pie(
                values=category_spending.values,
                names=category_spending.index,
                title="Spending by Category",
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Purples_r
            )

            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400
            )

            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            # Bar chart
            fig = px.bar(
                x=category_spending.index,
                y=category_spending.values,
                title="Category Breakdown",
                labels={'x': 'Category', 'y': 'Amount ($)'},
                color=category_spending.values,
                color_continuous_scale='Reds'
            )

            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=400,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

        # Category trends
        st.subheader("Category Trends")

        category_monthly = filtered_transactions[filtered_transactions['Type'] == 'expense'].copy(
        )
        category_monthly['Month'] = category_monthly['Date'].dt.to_period(
            'M').astype(str)
        category_trends = category_monthly.groupby(['Month', 'Category'])[
            'Amount'].sum().abs().reset_index()

        fig = px.line(
            category_trends,
            x='Month',
            y='Amount',
            color='Category',
            markers=True,
            title="Category Spending Trends"
        )

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=400
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Account Analysis")

        # Spending by account
        account_spending = filtered_transactions[
            filtered_transactions['Type'] == 'expense'
        ].groupby('Account')['Amount'].sum().abs()

        col_a, col_b = st.columns(2)

        with col_a:
            fig = px.pie(
                values=account_spending.values,
                names=account_spending.index,
                title="Spending by Account",
                color_discrete_sequence=px.colors.sequential.Blues_r
            )

            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=350
            )

            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            # Transaction count by account
            account_count = filtered_transactions.groupby(
                'Account').size().reset_index(name='Count')

            fig = px.bar(
                account_count,
                x='Account',
                y='Count',
                title="Transactions by Account",
                color='Count',
                color_continuous_scale='Greens'
            )

            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white'),
                height=350,
                showlegend=False
            )

            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Calendar View")

        # Heatmap
        calendar_data = filtered_transactions[filtered_transactions['Type'] == 'expense'].copy(
        )
        calendar_data['DayOfWeek'] = calendar_data['Date'].dt.day_name()
        calendar_data['Week'] = calendar_data['Date'].dt.isocalendar().week

        heatmap_data = calendar_data.groupby(['Week', 'DayOfWeek'])[
            'Amount'].sum().abs().reset_index()

        # Pivot for heatmap
        heatmap_pivot = heatmap_data.pivot(
            index='Week', columns='DayOfWeek', values='Amount').fillna(0)

        # Reorder columns
        day_order = ['Monday', 'Tuesday', 'Wednesday',
                     'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot = heatmap_pivot.reindex(
            columns=[d for d in day_order if d in heatmap_pivot.columns])

        fig = px.imshow(
            heatmap_pivot,
            labels=dict(x="Day of Week", y="Week", color="Amount ($)"),
            title="Spending Heatmap",
            color_continuous_scale='Reds',
            aspect='auto'
        )

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

        # Day of week analysis
        st.subheader("Day of Week Analysis")

        dow_spending = calendar_data.groupby('DayOfWeek')['Amount'].agg([
            'sum', 'mean', 'count']).abs()
        dow_spending = dow_spending.reindex(
            [d for d in day_order if d in dow_spending.index])

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Total Spending", "Average per Transaction")
        )

        fig.add_trace(
            go.Bar(x=dow_spending.index,
                   y=dow_spending['sum'], name='Total', marker_color='#ef4444'),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=dow_spending.index,
                   y=dow_spending['mean'], name='Average', marker_color='#6366f1'),
            row=1, col=2
        )

        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=350,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Origin Financial</strong> - AI-Powered Financial Planning</p>
    <p style='font-size: 12px;'>Built with Streamlit & Google Gemini AI • Your data stays local and secure</p>
    <p style='font-size: 11px; color: #444;'>User: @{os.getenv('USER', 'karanssandhu')} • Version 2.0 • Last updated: 2025</p>
</div>
""", unsafe_allow_html=True)
