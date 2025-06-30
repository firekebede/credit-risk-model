## Credit Scoring Business Understanding

### 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Accord requires banks to establish robust internal risk models to quantify and justify credit risk exposure. This pushes financial institutions to use models that are not just accurate, but also **interpretable, transparent, and auditable**. As a result, our credit scoring solution must be well-documented and explainable to satisfy both **regulatory compliance** and internal risk management. A black-box model without clarity or rationale would fail both internal audits and external regulatory reviews.

---

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In the absence of a direct label indicating customer default, we must define a **proxy variable** using observable behavior (e.g., poor RFM scores). This allows us to train a supervised model using approximations of risk. However, this introduces **label risk** and **bias**, which can lead to:

- Rejecting potentially good customers (false negatives),
- Approving high-risk customers (false positives),
- Unintended discrimination or model drift over time,
- Regulatory scrutiny if the proxy is not justifiable.

To mitigate this, the proxy must be defined based on **business logic**, validated frequently, and backed by documentation.

---

### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

| Model Type                            | Pros                                                                                             | Cons                                                       |
| ------------------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------------------------------- |
| **Logistic Regression with WoE**      | - Transparent and explainable<br>- Widely accepted by regulators<br>- Easy to monitor and update | - May underperform on complex, non-linear data             |
| **Gradient Boosting (e.g., XGBoost)** | - High predictive power<br>- Captures non-linear interactions well                               | - Difficult to interpret<br>- May face regulatory pushback |

In regulated environments, **interpretability is often prioritized**. Therefore, simple models may be preferred for deployment, while complex models can support internal decision-making or be enhanced with tools like SHAP for explainability.

---

# credit-risk-model
