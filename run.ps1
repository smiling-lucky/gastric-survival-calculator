Set-Location $PSScriptRoot
$env:STREAMLIT_SERVER_HEADLESS = "true"
$env:STREAMLIT_BROWSER_GATHER_USAGE_STATS = "false"
streamlit run app.py --server.headless true --server.port 8501 --browser.gatherUsageStats false
