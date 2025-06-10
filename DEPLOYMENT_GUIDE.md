# 🚀 Deployment Guide - Share Your Dashboard Easily

## 🎯 **Goal**: Get a public URL that anyone can access without coding knowledge

---

## 📋 **Step-by-Step Instructions**

### Phase 1: Prepare Your Repository

1. **Initialize Git** (run in your project folder):
   ```bash
   git init
   git add .
   git commit -m "Add Digital Sustainability Dashboard"
   ```

2. **Create GitHub Repository**:
   - Go to [github.com](https://github.com)
   - Click **"New repository"**
   - Name it: `digital-sustainability-dashboard`
   - Make it **Public** (required for free Streamlit Cloud)
   - **Don't** initialize with README (you already have files)
   - Click **"Create repository"**

3. **Connect and Push**:
   ```bash
   git remote add origin https://github.com/YOUR_USERNAME/digital-sustainability-dashboard.git
   git branch -M main
   git push -u origin main
   ```

### Phase 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

2. **Create New App**:
   - Click **"New app"**
   - Choose **"From existing repo"**
   - Repository: `YOUR_USERNAME/digital-sustainability-dashboard`
   - Branch: `main`
   - Main file path: `dashboard.py`
   - App URL: (optional custom name)

3. **Deploy**:
   - Click **"Deploy!"**
   - Wait 2-3 minutes for deployment
   - Get your public URL!

### Phase 3: Share with Users

✅ **Success!** Your dashboard is now live at:
`https://your-app-name.streamlit.app`

**Share this URL with anyone** - they can access it instantly from any device!

---

## 🔧 **Troubleshooting**

### Common Issues:

1. **"Module not found"**
   - Check `requirements.txt` is in your repository
   - Ensure all dependencies are listed

2. **"File not found"**
   - Ensure all CSV files are in the repository
   - Check file names match exactly

3. **App won't start**
   - Check dashboard.py for syntax errors
   - View logs in Streamlit Cloud dashboard

### If Deployment Fails:
```bash
# Test locally first
streamlit run dashboard.py
# If it works locally, it should work on Streamlit Cloud
```
