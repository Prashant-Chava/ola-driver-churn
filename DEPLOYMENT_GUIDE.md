# Streamlit Deployment Guide

## Issues Fixed

### 1. **File Path Issue (CRITICAL)**
- **Problem**: The app was using relative paths like `'ola_churn_model/ola_churn_xgb.pkl'` which don't work reliably on Streamlit Cloud.
- **Fix**: Updated to use absolute paths by determining the script directory at runtime:
  ```python
  SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
  MODEL_DIR = os.path.join(SCRIPT_DIR, 'ola_churn_model')
  ```

### 2. **Streamlit Configuration (RECOMMENDED)**
- **Created**: `.streamlit/config.toml`
- **Includes**: Theme colors, server settings, and logging configuration for production

### 3. **Deployment Optimization (OPTIONAL BUT HELPFUL)**
- **Created**: `.streamlitignore` to exclude unnecessary files from deployment

## Deployment Steps

### For Streamlit Cloud (Recommended)

1. **Push code to GitHub** with these files:
   - `app.py` (updated)
   - `requirements.txt`
   - `runtime.txt`
   - `.streamlit/config.toml` (new)
   - `ola_churn_model/` (all .pkl files)

2. **Create Streamlit account**: https://streamlit.io/cloud

3. **Deploy your app**:
   - Click "New app"
   - Select your GitHub repository
   - Set main file path: `app.py`
   - Click "Deploy"

### For Local Testing

```bash
streamlit run app.py
```

## Files to Include in Git

Make sure your `.gitignore` includes:
```
.streamlit/secrets.toml
__pycache__/
*.pyc
.venv/
```

But ensure these ARE committed:
- `ola_churn_model/*.pkl` — Include all model files
- `.streamlit/config.toml` — Deployment config
- `.streamlitignore` — Deployment optimization

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| "Model not found" error | ✓ Fixed with absolute paths |
| Styling not loading | Check if Google Fonts accessible (usually is on Cloud) |
| App timeout | Model load caching (`@st.cache_resource`) handles this |
| Dependencies error | Check `requirements.txt` versions match your environment |

## Testing Before Deployment

1. Test locally: `streamlit run app.py`
2. Verify model loads without errors
3. Test predictions with sample data
4. Check app styling renders correctly

---

**If still failing on Streamlit Cloud:**
- Check deployment logs (App → Manage app → View logs)
- Verify all `.pkl` files are committed to git
- Ensure `runtime.txt` specifies Python 3.11
