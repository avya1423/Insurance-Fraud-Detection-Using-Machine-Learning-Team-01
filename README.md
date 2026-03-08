# Insurance Fraud Detection Using Machine Learning

**Developed by:** Avya Nand  
**Institution:** LNCT, Bhopal  
**Program:** Smart Bridge Internship

## 📌 Project Overview
This project aims to detect fraudulent insurance claims using advanced Machine Learning algorithms. Insurance fraud is a significant challenge for the industry, leading to massive financial losses. By automating the detection process, we can identify suspicious patterns that might be missed by manual audits.

## 🛠️ Tech Stack
* **Language:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn (Decision Tree, Random Forest, KNN), XGBoost, Matplotlib, Seaborn
* **Web Framework:** Flask (for model deployment)
* **IDE:** Jupyter Notebook / VS Code

## 🏆 Certifications & Achievements
* Received the **AI Sashakt** badge from IndiaAI Mission, MeitY, and Intel India for taking the AI Responsibility Pledge.
* Participant in the **MY Bharat Budget Quest 2026**.

## 📖 Epic 1: Problem Understanding

### Activity 1.1: Specify the business problem
The insurance industry faces a massive challenge with fraudulent claims, which cost billions of dollars annually. Manual investigation of every claim is time-consuming, expensive, and prone to human error. The business problem is to develop an automated system that can accurately and quickly identify potentially fraudulent claims from legitimate ones, allowing human investigators to focus only on high-risk cases.

### Activity 1.2: Business requirements
To solve this problem effectively, the developed machine learning solution must meet the following requirements:
* **High Precision & Recall:** The model must minimize False Positives (flagging genuine claims as fraud) to avoid customer dissatisfaction, while maintaining a high Recall to catch actual frauds.
* **Automation:** The system should seamlessly integrate with existing claim processing workflows.
* **Scalability:** It must handle a large volume of daily insurance claims efficiently.
* **Interpretability:** The model should provide insights into why a claim was flagged as fraudulent.

### Activity 1.3: Literature Survey
Traditionally, insurance fraud detection relied on rule-based systems and manual audits, which fraudsters quickly learned to bypass. Recent advancements in Machine Learning have shifted the focus toward predictive modeling. Studies show that supervised learning algorithms—such as Decision Trees, Random Forests, K-Nearest Neighbors (KNN), and XGBoost—are highly effective in recognizing complex, hidden patterns in historical claim data that indicate fraudulent behavior.

### Activity 1.4: Social or Business Impact
* **Business Impact:** Successfully deploying this model will drastically reduce financial losses for the insurance company and cut down the operational costs associated with manual investigations.
* **Social Impact:** By saving money on fraudulent payouts, insurance companies can prevent premium hikes. This ensures that honest, everyday customers do not have to pay higher premiums to cover the costs of fraud, making insurance more affordable and fair for everyone.

## 📂 Folder Structure
```text
├── data/               # Contains dataset (e.g., insurance_claims.csv)
├── notebooks/          # Jupyter Notebooks for EDA & Model Training
├── models/             # Saved trained model files (.pkl)
├── static/             # CSS & Images for web UI
├── templates/          # HTML files for Flask app
├── app.py              # Main Flask application
├── requirements.txt    # List of project dependencies
└── README.md           # Project documentation
