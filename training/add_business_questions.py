"""Add business-style questions (English and Hinglish) to data/questions.csv."""
import pandas as pd

ENGLISH = [
    ("What does KPI stand for in business?", "Key Performance Indicator"),
    ("If a company's revenue is 500000 rupees and its expenses are 350000 rupees, what is the profit?", "150000 rupees"),
    ("What is the formula for profit margin percentage?", "(Profit / Revenue) x 100"),
    ("An employee earns 60000 rupees per month. What is the annual salary?", "720000 rupees"),
    ("What does SQL stand for?", "Structured Query Language"),
    ("Which SQL keyword removes duplicate rows from a SELECT result?", "DISTINCT"),
    ("What is the difference between a primary key and a foreign key?", "A primary key uniquely identifies each row in a table. A foreign key refers to the primary key of another table to link the two tables."),
    ("What does HR stand for in a company?", "Human Resources"),
    ("A product is sold at 1200 rupees after a 20% discount. What was the original price?", "1500 rupees"),
    ("What is GST in India?", "Goods and Services Tax, an indirect tax on the supply of goods and services."),
    ("What does ROI stand for and how is it calculated?", "Return on Investment = (Net profit / Cost of investment) x 100"),
    ("A team of 5 people each works 8 hours a day for 5 days. How many total person-hours is that?", "200 person-hours"),
    ("What does CRM stand for?", "Customer Relationship Management"),
    ("Sales grew from 80 lakh to 100 lakh rupees. What is the percentage growth?", "25%"),
    ("What is the purpose of a firewall in IT?", "It monitors and controls network traffic using security rules, blocking unauthorized access."),
    ("What does VPN stand for?", "Virtual Private Network"),
    ("What is the difference between a software bug and a feature request?", "A bug is a defect where the software behaves incorrectly. A feature request asks for new functionality."),
    ("Write a SQL query to count the rows in a table named orders.", "SELECT COUNT(*) FROM orders;"),
    ("What does the SQL clause GROUP BY do?", "It groups rows that share a value so aggregate functions like SUM or COUNT can be applied to each group."),
    ("What is an SLA?", "Service Level Agreement, a contract defining the expected level of service such as uptime or response time."),
    ("A company has 250 employees and 12% leave in a year. How many employees leave?", "30 employees"),
    ("What does EBITDA stand for?", "Earnings Before Interest, Taxes, Depreciation and Amortization"),
    ("What is two-factor authentication?", "A security method that requires two different proofs of identity, such as a password plus a one-time code."),
    ("A laptop costs 45000 rupees plus 18% GST. What is the total price?", "53100 rupees"),
    ("What is the difference between revenue and profit?", "Revenue is the total income from sales. Profit is what remains after subtracting expenses from revenue."),
]

HINGLISH = [
    ("Agar kisi company ki sales 200000 rupaye hai aur kharcha 150000 rupaye hai, to profit kitna hua?", "50000 rupees"),
    ("Ek employee ki monthly salary 40000 rupaye hai. Uski saalana salary kitni hogi?", "480000 rupees"),
    ("SQL mein duplicate rows hatane ke liye kaunsa keyword use karte hain?", "DISTINCT"),
    ("GST ka full form kya hai?", "Goods and Services Tax"),
    ("Ek product ki price 500 rupaye hai aur us par 10% discount mila. Final price kitni hogi?", "450 rupees"),
    ("Primary key aur foreign key mein kya difference hai?", "A primary key uniquely identifies each row in a table. A foreign key refers to the primary key of another table to link the two tables."),
    ("Ek team mein 10 log hain aur har koi 6 ghante roz kaam karta hai. 4 din mein total kitne ghante kaam hoga?", "240 hours"),
    ("KPI ka matlab kya hota hai?", "Key Performance Indicator, a metric used to measure performance."),
    ("Agar company ke 200 employees hain aur 15% resign kar dete hain, to kitne log gaye?", "30 employees"),
    ("Firewall ka kaam kya hota hai?", "It monitors and filters network traffic and blocks unauthorized access."),
    ("SQL mein orders table ki total rows count karne ki query likho.", "SELECT COUNT(*) FROM orders;"),
    ("Revenue aur profit mein kya fark hai?", "Revenue is the total income from sales. Profit is what remains after subtracting expenses from revenue."),
    ("Ek laptop 50000 rupaye ka hai, us par 18% GST lagta hai. Total kitna dena hoga?", "59000 rupees"),
    ("VPN ka full form kya hai?", "Virtual Private Network"),
    ("Sales 50 lakh se badhkar 60 lakh ho gayi. Growth percentage kitna hai?", "20%"),
]

df = pd.read_csv("data/questions.csv")
df = df[~df["source"].isin(["business", "business_hinglish"])]  # safe to rerun

rows = []
for i, (q, g) in enumerate(ENGLISH):
    rows.append({"id": f"biz_{i}", "source": "business", "question": q, "gold": g, "aliases": ""})
for i, (q, g) in enumerate(HINGLISH):
    rows.append({"id": f"bizhi_{i}", "source": "business_hinglish", "question": q, "gold": g, "aliases": ""})

df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
df.to_csv("data/questions.csv", index=False)
print("Total questions:", len(df))
print(df["source"].value_counts())