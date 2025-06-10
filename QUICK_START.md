# 🌍 Digital Sustainability Dashboard - Quick Access

## 📱 **For Non-Technical Users**

### Option 1: Online Access (Simplest)
**Just click the link below - no installation needed!**

🔗 **Dashboard URL**: *[Will be added after deployment]*

---\

### Option 2: Run Locally (If you're comfortable with basic commands)

#### Requirements:
- Python 3.8+ installed on your computer
- 5 minutes of setup time

#### Steps:
1. **Download the files**
   - Download all files from this repository
   - Extract to a folder on your computer

2. **Open Terminal/Command Prompt**
   - Windows: Press `Win + R`, type `cmd`, press Enter
   - Mac: Press `Cmd + Space`, type `Terminal`, press Enter

3. **Navigate to the folder**
   ```bash
   cd path/to/your/downloaded/folder
   ```

4. **Set up the environment**
   ```bash
   python -m venv thesis_env
   source thesis_env/bin/activate  # Mac/Linux
   # OR
   thesis_env\Scripts\activate     # Windows
   ```

5. **Install requirements**
   ```bash
   pip install -r requirements.txt
   ```

6. **Run the dashboard**
   ```bash
   streamlit run dashboard.py
   ```

7. **Open your browser**
   - Go to: `http://localhost:8501`

---

## 📊 **How to Use the Dashboard**

### Navigation
Use the **sidebar** on the left to switch between different views:

1. **📊 Overview** - See overall maturity scores and radar charts
2. **🔍 Detailed Analysis** - Deep dive into specific dimensions
3. **📈 Gap Analysis** - Identify priority improvement areas
4. **💡 Recommendations** - Get AI-driven suggestions
5. **🔍 Data Explorer** - Inspect raw data (for debugging)

### Understanding the Scores
- **Scale**: 1-5 (1 = Basic, 5 = Advanced/Autonomous)
- **Dimensions**: People, Process, Policy
- **ESG Categories**: Environmental, Social, Governance
- **Time Frames**: Current State vs Future Goals

### Key Metrics
- **Maturity Gap**: Difference between current and target states
- **Priority Areas**: Where to focus improvement efforts first
- **Progress Rate**: How much of your maturity journey is complete

---

## 🆘 **Need Help?**

### Common Issues
1. **Nothing loads**: Check your internet connection
2. **Errors in local setup**: Ensure Python 3.8+ is installed
3. **Data not showing**: Verify CSV files are in the same folder

### Contact
For questions or issues, contact: [Your Email Here]

---

*This dashboard helps organizations assess their digital sustainability maturity across Environmental, Social, and Governance dimensions.* 