import re

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

replacements = [
    (r'rgba\(255,\s*255,\s*255,\s*0\.06\)', 'var(--border-light)'),
    (r'rgba\(255,\s*255,\s*255,\s*0\.07\)', 'var(--border-light)'),
    (r'rgba\(255,\s*255,\s*255,\s*0\.08\)', 'var(--border-light)'),
    (r'rgba\(255,\s*255,\s*255,\s*0\.1\)', 'var(--border-med)'),
    (r'rgba\(255,\s*255,\s*255,\s*0\.02\)', 'var(--bg-hover)'),
    (r'rgba\(255,\s*255,\s*255,\s*0\.03\)', 'var(--bg-hover)'),
    (r'rgba\(255,\s*255,\s*255,\s*0\.05\)', 'var(--border-light)'),
    (r'rgba\(0,\s*0,\s*0,\s*0\.5\)', 'var(--shadow-str)'),
]

for old, new in replacements:
    content = re.sub(old, new, content)

# Let's add var(--bg-hover) and var(--shadow-str) to :root
old_light = """            --border-med: rgba(59, 130, 246, 0.5);
        }"""
new_light = """            --border-med: rgba(59, 130, 246, 0.5);
            --bg-hover: rgba(0,0,0,0.03);
            --shadow-str: rgba(0,0,0,0.1);
        }"""

old_dark = """            --border-med: rgba(255,255,255,0.15);
        }"""
new_dark = """            --border-med: rgba(255,255,255,0.15);
            --bg-hover: rgba(255,255,255,0.03);
            --shadow-str: rgba(0,0,0,0.5);
        }"""

content = content.replace(old_light, new_light)
content = content.replace(old_dark, new_dark)

with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)
