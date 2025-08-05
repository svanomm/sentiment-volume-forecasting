# Assessing the Usefulness of News Sentiment for Real-Time Airline Stock Prediction

A project by Steven VanOmmeren examining the impact of news sentiment on airline stocks using advanced machine learning techniques and real-time data analysis.

## Overview

This research project leverages the Global Database of Events, Language, and Tone (GDELT) to analyze how news sentiment affects airline stock volumes in near-real-time. The commercial airline industry is unique in that adverse events (crashes, incidents, etc.) are highly publicized and can dramatically impact public trust and stock prices.

### Key Features

- **Real-time Analysis**: Predicts stock price changes in 15-minute increments during trading days
- **Large-scale Data**: Analyzes sentiment from 1.3 million news articles mentioning major U.S. airlines
- **Comprehensive Coverage**: Focuses on 7 major U.S. commercial airlines from January 2018 to May 2025

### Research Objectives

1. Examine the impact of adverse news events on airline stock prices at the near-real-time level
2. Identify and analyze adverse news events using GDELT data
3. Predict real-time stock volumes and price changes more accurately than existing models
4. Demonstrate the economic value of GDELT for business monitoring applications

## Data Sources

- **GDELT (Global Database of Events, Language, and Tone)**: Real-time news sentiment data
- **Stock Market Data**: Real-time stock prices and volumes for major U.S. airlines

### Prerequisites

- Python 3.12
- `uv` package manager (recommended) or `pip`

## Key Findings

This research demonstrates:

- **Superior Performance**: Models achieve better accuracy than state-of-the-art approaches in predicting stock volumes
- **Real-time Capability**: System can process and predict based on news sentiment in near-real-time
- **Economic Value**: GDELT data provides actionable insights for business monitoring
- **Scalability**: Framework can be adapted to other industries beyond airlines

## Academic Paper

The complete research findings are documented in a formal academic paper located in the `Paper/` directory. The paper includes:

- Comprehensive literature review
- Detailed methodology
- Statistical analysis and results
- Economic implications and applications

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{vanommeren2025airline,
  title={Predicting Intraday Trading Volume with News Sentiment: An Analysis of U.S. Airline Stocks},
  author={VanOmmeren, Steven},
  year={2025},
  url={[https://github.com/svanomm/svo-directed-practicum](https://github.com/svanomm/sentiment-volume-forecasting)}
}
```

## License

This project is licensed under the MIT License, however we make no claim as to the licenses of the packages relied upon in this work. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration opportunities, please contact Steven VanOmmeren.

---

*This project appears to be the first to publicly examine such a large volume of real-time GDELT data for airline stock prediction, providing valuable insights for both academic research and practical business applications.*
