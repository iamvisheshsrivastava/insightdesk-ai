# Append simple SPA catch-all route
import os
os.chdir(r'c:\Users\sriva\Desktop\Projects\Personal Repositories\insightdesk-ai')

code = '''

# ===== Final SPA Catch-All Route =====
@app.get('/{full_path:path}')
async def serve_frontend(full_path: str):
    """Serve frontend index.html for SPA routing."""
    from pathlib import Path
    index_file = Path('frontend/dist/index.html')
    if index_file.exists():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail='Frontend not found')
'''

with open('src/api/main.py', 'a') as f:
    f.write(code)
print('✅ SPA catch-all added to main.py')
