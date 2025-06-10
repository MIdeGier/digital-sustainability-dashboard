# 🌍 Digital Sustainability Maturity Dashboard

A comprehensive Streamlit dashboard for assessing and visualizing digital sustainability maturity across Environmental, Social, and Governance (ESG) dimensions.

## 📋 Overview

This dashboard helps organizations assess their digital maturity in sustainability initiatives by analyzing survey responses across three key dimensions:

- **People**: Skills, knowledge, and competencies
- **Process**: Equipment, production, and operational intelligence  
- **Policy**: Strategy, formalization, and governance integration

Each dimension is evaluated across ESG categories:
- **Environmental**: Energy consumption, GHG emissions, air pollution, waste management, etc.
- **Social**: Diversity & inclusion, training & skills, health & safety, work-life balance
- **Governance**: Ethical culture & behavior

## ✨ Features

### 📊 Multi-View Dashboard
- **Overview**: High-level radar charts and key performance metrics
- **Detailed Analysis**: Dimension-specific breakdowns with ESG scoring
- **Gap Analysis**: Visual identification of improvement priorities
- **Recommendations**: AI-driven suggestions for digital maturity advancement
- **Data Explorer**: Raw data inspection and column structure analysis

### 📈 Advanced Visualizations
- Interactive radar charts comparing current state vs. future goals
- Gap analysis bar charts with priority coding
- Progress indicators and metric cards
- Color-coded data tables with background gradients

### 🔍 Intelligent Data Processing
- Automatic survey response parsing and categorization
- Smart text analysis to map responses to maturity levels (1-5)
- ESG impact extraction from question text
- Robust error handling and data validation

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. **Clone or download the project files**
   ```bash
   cd your-project-directory
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv thesis_env
   source thesis_env/bin/activate  # On Windows: thesis_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the dashboard**
   ```bash
   streamlit run dashboard.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501` to view the dashboard

## 📁 Project Structure

```
Thesis/
├── dashboard.py              # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # Project documentation
├── Survey_data.csv          # Survey response data
├── DST_Excel_QAweights.csv  # Question weights and maturity descriptions
├── DST_Excel_Complexity.csv # Additional complexity data
├── DST_Excel_ESRS2Tech.csv  # ESRS to technology mapping
├── DST_Excel_Tech2ESRS.csv  # Technology to ESRS mapping
└── DST_Excel_TradeOffs.csv  # Trade-offs analysis data
```

## 📊 Data Format

### Survey Data Structure
The `Survey_data.csv` file should contain:
- **Basic Info**: ID, timestamps, contact information
- **Sustainability Importance**: Ratings for Environmental, Social, Governance priorities
- **Inclusion Questions**: Yes/No decisions for impact categories
- **Weight Questions**: Importance ratings (1-4 scale)
- **Maturity Questions**: Detailed responses about current and future states

### Maturity Level Mapping
Responses are automatically mapped to 5 maturity levels:

1. **Level 1**: Basic, manual processes with limited digital tools
2. **Level 2**: Dashboard-based monitoring and diagnostic capabilities
3. **Level 3**: Predictive analytics and proactive interventions
4. **Level 4**: Prescriptive systems with automated recommendations
5. **Level 5**: Autonomous AI-driven optimization and adaptation

## 🎯 Key Features Explained

### Radar Visualizations
- **Current State**: Orange visualization showing present maturity levels
- **Future Goals**: Blue visualization displaying target aspirations
- **Comparative Analysis**: Easy identification of gaps and priorities

### Gap Analysis
- Automatic calculation of improvement areas
- Priority classification (High/Medium/Low)
- Visual representation with color-coded bar charts

### Smart Recommendations
Context-aware suggestions based on:
- Identified maturity gaps
- ESG category priorities
- Dimensional focus areas
- Industry best practices

## 🔧 Configuration & Customization

### Adding New ESG Categories
Modify the `esg_mapping` dictionary in `calculate_maturity_scores()`:

```python
esg_mapping = {
    'your_new_category': 'E',  # or 'S' or 'G'
    # ... existing mappings
}
```

### Customizing Maturity Level Detection
Update the `level_indicators` dictionary in `map_response_to_maturity()`:

```python
level_indicators = {
    5: ['your', 'level5', 'keywords'],
    4: ['your', 'level4', 'keywords'],
    # ... etc
}
```

### Styling Customization
Modify the CSS in the `st.markdown()` section at the top of `dashboard.py`.

## 🐛 Troubleshooting

### Common Issues

1. **"No valid maturity scores could be calculated"**
   - Check that your CSV file has the correct column structure
   - Ensure survey responses contain recognizable maturity level indicators
   - Use the Data Explorer to debug column parsing

2. **Missing visualizations**
   - Verify all required data files are present
   - Check that survey responses are not empty or null
   - Review the maturity level mapping logic

3. **Performance issues**
   - Large datasets may require data sampling
   - Consider caching optimizations for repeated calculations

### Data Quality Checks
The dashboard includes built-in validation:
- Automatic detection of missing or malformed data
- Column structure analysis and reporting
- Response completion rate calculations

## 📈 Future Enhancements

Potential improvements and extensions:

- [ ] **Multi-organization comparison** capabilities
- [ ] **Historical trend analysis** for tracking progress over time
- [ ] **Benchmark scoring** against industry standards
- [ ] **Export functionality** for reports and presentations
- [ ] **Advanced filtering** by organization size, industry, region
- [ ] **Machine learning predictions** for maturity trajectory
- [ ] **Integration with ESG reporting** frameworks

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your improvements
4. Test thoroughly with sample data
5. Submit a pull request with detailed description

## 📄 License

This project is part of a thesis research initiative. Please contact the authors for usage permissions and licensing information.

## 👥 Authors

- **Research Team**: Digital Sustainability Maturity Assessment
- **Institution**: [Your Institution Name]
- **Contact**: [Your Contact Information]

## 🙏 Acknowledgments

- Survey participants who provided valuable maturity assessment data
- Research advisors and methodology consultants
- Open-source community for excellent visualization libraries

---

**Built with**: Streamlit • Plotly • Pandas • NumPy
