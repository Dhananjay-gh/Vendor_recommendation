import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Change default theme to 'light'
content = content.replace("st.session_state.get('theme', 'dark')", "st.session_state.get('theme', 'light')")
content = content.replace("st.session_state.get('theme', \"dark\")", "st.session_state.get('theme', 'light')")

# 2. Add toggle to Landing Page (render_landing_page)
landing_start = """def render_landing_page():
    # ── THEME TOGGLE ──
    col_empty, col_toggle = st.columns([8, 1])
    with col_toggle:
        is_light = st.session_state.get('theme', 'light') == 'light'
        if st.toggle("☀ Light Mode", value=is_light, key="landing_theme"):
            if st.session_state.get('theme') != 'light':
                st.session_state['theme'] = 'light'
                st.rerun()
        else:
            if st.session_state.get('theme') != 'dark':
                st.session_state['theme'] = 'dark'
                st.rerun()
"""
if "# ── THEME TOGGLE ──" not in content.split('def render_landing_page():')[1][:200]:
    content = content.replace('def render_landing_page():', landing_start)

# 3. Fix missing hardcoded backgrounds in CSS/HTML
replacements = [
    (r'rgba\(10,\s*14,\s*26,\s*0\.85\)', 'var(--bg-card)'),
    (r'rgba\(10,\s*16,\s*32,\s*0\.7\)', 'var(--bg-card)'),
    (r'rgba\(12,\s*18,\s*36,\s*0\.9\)', 'var(--bg-card)'),
    (r'rgba\(6,\s*10,\s*24,\s*0\.9\)', 'var(--bg-card)'),
    (r'#0a1020', 'var(--bg-card)'),
]

for old, new in replacements:
    content = re.sub(old, new, content)

# 4. Update the light theme variables to have blue borders and pure white cards
# Let's find the css_vars block for light mode and replace the borders.
old_light_borders = """            --border-light: rgba(0,0,0,0.08);
            --border-med: rgba(0,0,0,0.15);"""
new_light_borders = """            --border-light: rgba(59, 130, 246, 0.2);
            --border-med: rgba(59, 130, 246, 0.5);"""
content = content.replace(old_light_borders, new_light_borders)

# Also ensure plot background in light mode is pure white, not transparent-ish
content = content.replace("st.session_state['plot_bg'] = 'rgba(255,255,255,0.8)'", "st.session_state['plot_bg'] = '#ffffff'")

# The chat user messages have `background: rgba(80, 160, 255, 0.1); border: 1px solid rgba(80, 160, 255, 0.3); color: #c8d8f0;`
# We should change `#c8d8f0` to `var(--text-main)` in those inline styles.
content = content.replace('color: #c8d8f0;', 'color: var(--text-main);')

# The vendor cards on the left have `border: 1px solid rgba(255, 255, 255, 0.15)`
# We already replaced the regex for rgba(255,255,255,0.15), it should be var(--border-med).

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
