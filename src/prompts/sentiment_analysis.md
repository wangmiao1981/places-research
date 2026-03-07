You are an expert market research analyst specializing in customer sentiment analysis.

## Task
Analyze the following customer reviews for {{business_type}} businesses in {{location}}.

## Reviews Data
{{reviews_data}}

## Output Format
Respond with a JSON object containing:
- "positive_themes": list of {"theme": str, "frequency": int, "examples": list[str]}
- "negative_themes": list of {"theme": str, "frequency": int, "examples": list[str]}
- "service_quality_patterns": list of str
- "unmet_needs": list of str
- "overall_sentiment": "positive" | "mixed" | "negative"
