# Consumer Complaint Priority Scoring API

## 1. Business Context

Consumer protection agencies and financial institutions receive hundreds of thousands of consumer complaints every year. These complaints vary widely in severity:
- Some are informational or low impact
- Others represent high-risk cases that may require urgent attention due to:
    - unresolved disputes
    - delayed or missing company responses
    - potential regulatory escalation

Manually reviewing and prioritizing complaints is time-consuming, inconsistent, and difficult to scale.

**Business Goal**

The goal of this project is to automatically prioritize consumer complaints based on their narrative text, helping teams focus attention on cases that are more likely to require escalation.

Instead of producing only a binary decision, the system outputs a probability score that can be used to:

- rank complaints by risk
- tune prioritization thresholds based on operational capacity
- balance false positives vs. missed escalations


## 2. Problem Framing

This is framed as a binary classification problem:

`Given a complaint narrative, what is the probability that the complaint is high priority?`

**Target Definition**

A complaint is labeled as high priority if either:

- the company did not respond in a timely manner
- the consumer explicitly disputed the company’s response

This definition reflects operational and regulatory risk, not just customer sentiment.


## 3. Data

The data used for this project was sourced from the CFPB Consumer Complaint Database, which is publicly available. It was filtered to include only Credit Card complaints with published consumer narratives, downloaded on January 21, 2026.

**Data collection**

- ~100,000 complaint narratives
- Highly imbalanced target (high priority ≈ 4–5%)
- Free-form natural language text

**Why this data is appropriate**

- Real-world, business-relevant complaints
- No synthetic labeling
- Representative of complaint intake systems used by regulators and financial institutions


## 4. Modeling Approach

I used TF-IDF vectorization with unigrams and bigrams to capture context beyond single words. To keep the model lean and performant, I removed standard English stopwords and capped vocabulary size, ensuring a manageable feature matrix without sacrificing predictive power.

For the classification task itself, I opted for a Logistic Regression model equipped with class weighting. Given that the dataset is naturally imbalanced, adjusting the weights allows the model to pay closer attention to the minority "high priority" cases. I chose this specific algorithm because it serves as an incredibly robust baseline for text classification; it is highly interpretable, provides well-calibrated probability scores, and offers the lightning-fast inference speeds required for production environments.


## 5. Evaluation Strategy

When dealing with highly imbalanced data, traditional metrics like accuracy can be misleading—a model could be 99% accurate simply by guessing "normal priority" every time. To get a true sense of performance, I focus on ROC AUC to measure the model’s overall ability to rank cases correctly.

More importantly, I use Precision@10% to mirror how the system would be used in a real-world operational setting. If a team of reviewers only has the capacity to examine the top 10% of cases flagged as "most risky," we need to know exactly what proportion of those cases are truly high priority. This metric ensures the model is providing tangible value to the people using it.

**Current performance snapshot**
|Metric|Result|
|-|-|
|ROC-AUC|0.857|
|Precision@10%|0.211|
|Training time|112.98ms|

## 6. Decision Threshold

The model generates a continuous probability score rather than a simple "yes" or "no" answer. By default, a decision threshold of 0.5 is applied to convert this score into a binary label: 1 for high priority and 0 for normal priority.

By separating the raw scoring from the final threshold, the system becomes significantly more flexible. This architecture allows for ranked review workflows—where humans can tackle the highest scores first—and enables stakeholders to adjust the system's sensitivity (making it more or less "aggressive") without needing to retrain the entire model.

---

## 7. System Architecture

The system is split into two distinct phases to ensure reliability and ease of deployment:

**Training (Offline)**

The training process happens outside of the containerized environment. It involves a dedicated pipeline that handles data preprocessing, automated labeling, and the training of the combined TF-IDF and Logistic Regression pipeline. Once the model is optimized, the entire state is serialized and saved as a pipeline.joblib artifact.

**Inference (Online)**

For real-time use, a FastAPI service serves as the primary interface. Upon startup, the service loads the pre-trained joblib file and exposes a /score endpoint. When a request is received, the API returns a structured response containing:
- The raw probability score.
- The predicted label based on the threshold.
- The specific threshold used to make the determination.

## 8. Project Structure
```
complaint-priority/
├── model/
│   └── pipeline.joblib
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── processing.py
│   └── model.py
│
├── scripts/
│   └── train.py
│
├── main.py
├── Dockerfile
├── fly.toml
├── pyproject.toml
├── uv.lock
├── .dockerignore
├── .gitignore
└── README.md
```

## 9. How to Run Locally
### 9.1 Create environment and install dependencies
```bash
$ uv venv --python python3.12
$ source .venv/bin/activate
$ uv sync
```
### 9.2 Train the model

Download the CFPB complaints CSV from the [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) and place it in the data/ directory.
```bash
$ python -m scripts.train --input data/raw_data.csv
```

This produces:
```
model/pipeline.joblib
```
### 9.3 Run the API locally
```bash
$ uvicorn main:app --reload
```
### 9.4 Test the API
```bash
$ curl -X POST "http://localhost:8000/score" \
  -H "Content-Type: application/json" \
  -d '{"complaint_text":"I disputed the charges and the company failed to respond"}'
```

Example response:
```json
{
  "high_priority_score": 0.34,
  "predicted_label": 0,
  "threshold": 0.5
}
```

## 10. Containerization
The project follows a production-first Docker design, prioritizing stability and efficiency. A key architectural decision is that the Docker image is built for inference only. By keeping the heavy training process outside of the container, we ensure that the image remains lightweight and that the production environment is strictly for serving predictions. During the build process, the pre-trained model artifact is copied directly into the image, creating a portable, immutable environment that can be deployed anywhere.

To get the service running locally, you can build and launch the container using the following commands:

**Build the image:**
```bash
$ docker build -t complaint-priority .
```
**Run the container:**
```bash
$ docker run -p 8080:8080 complaint-priority
```
## 11. Deployment (Fly.io)
For hosting, I chose Fly.io because of its developer-friendly ecosystem and native support for Docker. It offers a generous free tier and, perhaps most importantly, a "scale-to-zero" feature that prevents billing when the service is idle. This makes it an ideal choice for projects where cost-efficiency is a priority.

The deployment process is streamlined through a simple CLI workflow:
```bash
$ flyctl launch
$ flyctl deploy
```

The service is configured to listen on 0.0.0.0 and dynamically respects Fly’s specific port environment variables. Once live, you can send requests to the public endpoint.

Live Endpoint: `POST https://cfpb-complaints-priority-scorer.fly.dev`

**Example Request:**
```json
{
  "complaint_text": "The company never responded and I escalated the dispute"
}
```

## 12. Key Takeaways

The core philosophy of this project is to prioritize business impact over model complexity. Rather than chasing the latest high-overhead architectures, this system utilizes interpretable, production-safe NLP techniques that provide clear value immediately.

The project demonstrates the complete end-to-end Machine Learning lifecycle—taking raw data through modeling and into a containerized cloud API. By maintaining a clear separation between training, inference, and deployment, the system remains modular and easy to maintain.

## 13. Future Improvements

While the current version provides a strong baseline, there are several paths for future iteration. I plan to refine the priority definitions to be more nuanced and introduce active learning, where the model can learn from real-time feedback provided by human reviewers.

Furthermore, moving from binary classification to multi-class prioritization would allow for more granular sorting of complaints. Finally, I intend to implement model explainability features to show reviewers the specific terms (such as "dispute" or "escalated") that contributed most to a high-priority score, building greater trust in the model's decisions.

## 14. Data Attribution & License

This project utilizes the Consumer Complaint Database provided by the Consumer Financial Protection Bureau (CFPB), a department of the US Federal Government.

Citation:

    Consumer Financial Protection Bureau. (2026). Consumer Complaint Database [Data set]. Retrieved January 21, 2026, from https://www.consumerfinance.gov/data-research/consumer-complaints/

License Note: The data is provided by a US government agency and is generally considered to be in the public domain within the United States. This software is intended for educational and demonstrative purposes. See the [LICENSE](LICENSE) file for details.