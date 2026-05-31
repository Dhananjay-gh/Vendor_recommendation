import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. CSS Variable Replacements for HTML strings
replacements = [
    (r'#050810', 'var(--bg-app)'),
    (r'#0a0e1a', 'var(--bg-card)'),
    (r'#c8d8f0', 'var(--text-main)'),
    (r'#f0f4ff', 'var(--text-head)'),
    (r'#94a3b8', 'var(--text-mute)'),
    (r'rgba\(255,\s*255,\s*255,\s*0\.05\)', 'var(--border-light)'),
    (r'rgba\(255,\s*255,\s*255,\s*0\.15\)', 'var(--border-med)'),
    (r'rgba\(255,\s*255,\s*255,\s*0\.1\)', 'var(--border-med)'),
]

# We need to NOT replace the colors inside Python code for Plotly!
# But Plotly uses dictionary values, e.g., 'color': '#c8d8f0'
# Let's fix Plotly first.

content = content.replace("plot_bgcolor='rgba(10,14,26,0.6)'", "plot_bgcolor=st.session_state.get('plot_bg', 'rgba(10,14,26,0.6)')")
content = content.replace("plot_bgcolor='rgba(10, 14, 26, 0.6)'", "plot_bgcolor=st.session_state.get('plot_bg', 'rgba(10,14,26,0.6)')")

content = content.replace("'color': '#c8d8f0'", "'color': st.session_state.get('text_main', '#c8d8f0')")
content = content.replace("color='#c8d8f0'", "color=st.session_state.get('text_main', '#c8d8f0')")
content = content.replace("color: '#c8d8f0'", "color: st.session_state.get('text_main', '#c8d8f0')")
content = content.replace('color="#c8d8f0"', 'color=st.session_state.get("text_main", "#c8d8f0")')

# Now apply CSS var replacements
for old, new in replacements:
    content = re.sub(old, new, content)

# 2. Inject CSS Variables into setup_analytics_styles
style_def = """def setup_analytics_styles():
    theme = st.session_state.get('theme', 'dark')
    
    if theme == 'light':
        css_vars = \"\"\"
        :root {
            --bg-app: #f8fafc;
            --bg-card: #ffffff;
            --text-main: #334155;
            --text-head: #0f172a;
            --text-mute: #64748b;
            --border-light: rgba(0,0,0,0.08);
            --border-med: rgba(0,0,0,0.15);
        }
        \"\"\"
        st.session_state['plot_bg'] = 'rgba(255,255,255,0.8)'
        st.session_state['text_main'] = '#334155'
    else:
        css_vars = \"\"\"
        :root {
            --bg-app: #050810;
            --bg-card: #0a0e1a;
            --text-main: #c8d8f0;
            --text-head: #f0f4ff;
            --text-mute: #94a3b8;
            --border-light: rgba(255,255,255,0.05);
            --border-med: rgba(255,255,255,0.15);
        }
        \"\"\"
        st.session_state['plot_bg'] = 'rgba(10,14,26,0.6)'
        st.session_state['text_main'] = '#c8d8f0'
        
    st.markdown(f"<style>{css_vars}</style>", unsafe_allow_html=True)
    st.markdown(\"\"\"
    <style>"""

content = content.replace('def setup_analytics_styles():\n    st.markdown("""\n    <style>', style_def)

# 3. Add Toggle button in main() or render_analytics_page
# Let's add it in render_analytics_page at the very top
analytics_page_start = """def render_analytics_page():
    # ── THEME TOGGLE ──
    col_empty, col_toggle = st.columns([8, 1])
    with col_toggle:
        is_light = st.session_state.get('theme', 'dark') == 'light'
        if st.toggle("☀ Light Mode", value=is_light):
            if st.session_state.get('theme') != 'light':
                st.session_state['theme'] = 'light'
                st.rerun()
        else:
            if st.session_state.get('theme') != 'dark':
                st.session_state['theme'] = 'dark'
                st.rerun()
"""
content = content.replace('def render_analytics_page():', analytics_page_start)


with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
