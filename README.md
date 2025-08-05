# Assessing the Usefulness of News Sentiment for Real-Time Airline Stock Prediction

A directed practicum project by Steven VanOmmeren examining the impact of news sentiment on airline stocks using advanced machine learning techniques and real-time data analysis.

## Overview

This research project leverages the Global Database of Events, Language, and Tone (GDELT) to analyze how news sentiment affects airline stock prices in near-real-time. The commercial airline industry is unique in that adverse events (crashes, incidents, etc.) are highly publicized and can dramatically impact public trust and stock prices.

### Key Features

- **Real-time Analysis**: Predicts stock price changes in 15-minute increments during trading days
- **Large-scale Data**: Analyzes sentiment from 1.3 million news articles mentioning major U.S. airlines
- **Advanced ML Models**: Utilizes RNN, LSTM, CNN, LightGBM, and other state-of-the-art techniques
- **Comprehensive Coverage**: Focuses on 7 major U.S. commercial airlines from January 2018 to May 2025
- **Multi-source Integration**: Combines GDELT, BLS, and stock market data

### Research Objectives

1. Examine the impact of adverse news events on airline stock prices at the near-real-time level
2. Identify and analyze adverse news events using GDELT data
3. Predict real-time stock volumes and price changes more accurately than existing models
4. Demonstrate the economic value of GDELT for business monitoring applications

## Data Sources

- **GDELT (Global Database of Events, Language, and Tone)**: Real-time news sentiment data
- **BLS (Bureau of Labor Statistics)**: Economic indicators and employment data
- **Stock Market Data**: Real-time stock prices and volumes for major U.S. airlines
- **News Articles**: 1.3 million articles mentioning airlines for sentiment analysis

## Project Structure

```
├── Scripts/
│   ├── 00 BLS Data/          # Bureau of Labor Statistics data processing
│   ├── 01 GDELT Data/        # GDELT data extraction and processing
│   ├── 02 Stocks Data/       # Stock market data collection
│   ├── 03 Combined Data/     # Data integration and merging
│   ├── 04 Basic Analysis/    # Exploratory data analysis and event studies
│   ├── 05 Models/           # Machine learning models and predictions
│   └── _archive/            # Archived scripts and utilities
├── Data/
│   ├── Raw/                 # Original data files
│   └── Processed/           # Cleaned and processed datasets
├── Paper/                   # LaTeX paper and bibliography
├── requirements.txt         # Python dependencies
└── requirements_no_gpu.txt  # CPU-only dependencies
```

## Setup Instructions

### Prerequisites

- Python 3.12
- `uv` package manager (recommended) or `pip`

### Environment Setup

1. **Install uv** (if not already installed):
   ```bash
   # Follow instructions at: https://docs.astral.sh/uv/getting-started/installation/
   ```

2. **Clone the repository**:
   ```bash
   git clone https://github.com/svanomm/svo-directed-practicum.git
   cd svo-directed-practicum
   ```

3. **Create and activate virtual environment**:
   ```bash
   # Create venv with Python 3.12
   uv venv --python 3.12
   
   # Activate the venv
   .venv/scripts/activate  # Windows
   source .venv/bin/activate  # macOS/Linux
   ```

4. **Install dependencies**:
   ```bash
   # For GPU support (recommended)
   uv pip install -r requirements.txt
   
   # For CPU-only installation
   uv pip install -r requirements_no_gpu.txt
   ```

## Usage

### Running the Analysis

The analysis is organized in numbered directories corresponding to the data processing pipeline:

1. **Data Collection**: Start with Scripts/00-02 to gather BLS, GDELT, and stock data
2. **Data Integration**: Use Scripts/03 to combine datasets
3. **Exploratory Analysis**: Run Scripts/04 for basic analysis and event identification
4. **Model Training**: Execute Scripts/05 to train and evaluate ML models

### Key Notebooks

- `Scripts/04 Basic Analysis/01 Summary Tables and Charts.ipynb` - Overview of data and key statistics
- `Scripts/04 Basic Analysis/02 Identifying Events.ipynb` - Event detection and analysis
- `Scripts/05 Models/05 Comparing Model Results.ipynb` - Model performance comparison

### Example: Running Basic Analysis

```bash
# Ensure virtual environment is activated
jupyter notebook "Scripts/04 Basic Analysis/01 Summary Tables and Charts.ipynb"
```

## Machine Learning Models

The project implements and compares several advanced models:

- **Traditional Models**: Baseline statistical models for comparison
- **Neural Networks**: RNN, LSTM, and CNN architectures for sequence prediction
- **LightGBM**: Gradient boosting models with hyperparameter tuning
- **Ensemble Methods**: Combining multiple models for improved accuracy

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
  title={Assessing the Usefulness of News Sentiment for Real-Time Airline Stock Prediction},
  author={VanOmmeren, Steven},
  year={2025},
  url={https://github.com/svanomm/svo-directed-practicum}
}
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or collaboration opportunities, please contact Steven VanOmmeren.

---

*This project appears to be the first to publicly examine such a large volume of real-time GDELT data for airline stock prediction, providing valuable insights for both academic research and practical business applications.*
